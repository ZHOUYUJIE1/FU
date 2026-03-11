import copy
import random
import time
import torch
import numpy as np
from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.clients.clientfedau import clientFEDAUSCAFFOLD
from flcore.servers.serverbase import Server
from threading import Thread


class SCAFFOLD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        client_cls = clientFEDAUSCAFFOLD if getattr(args, "forget_strategy", "") == "fedau" else clientSCAFFOLD
        self.set_clients(client_cls)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.server_learning_rate = args.server_learning_rate
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))

        # 添加排除客户端功能（用于恢复阶段/重训练阶段）
        self.excluded_client_ids = []
        if getattr(args, "forget_strategy", "") == "fedau":
            self.force_include_client_id = getattr(args, "target_client_id", None)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # 如果设置了要排除的客户端，则在采样阶段排除它们
            if getattr(self, "excluded_client_ids", []):
                self.selected_clients = self.select_clients_with_exclusion()
            else:
                self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.after_aggregation()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientSCAFFOLD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model, self.global_c)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # 计算实际可用的客户端数量
        available_clients_count = len(self.selected_clients)
        # 确保采样数量不超过可用客户端数量
        sample_count = min(int((1-self.client_drop_rate) * available_clients_count), available_clients_count)
        
        if sample_count <= 0:
            sample_count = available_clients_count  # 如果计算结果为0或负数，使用所有可用客户端
        
        active_clients = random.sample(self.selected_clients, sample_count)

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        # self.delta_ys = []
        # self.delta_cs = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.delta_ys.append(client.delta_y)
                # self.delta_cs.append(client.delta_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        # # original version
        # for dy, dc in zip(self.delta_ys, self.delta_cs):
        #     for server_param, client_param in zip(self.global_model.parameters(), dy):
        #         server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
        #     for server_param, client_param in zip(self.global_c, dc):
        #         server_param.data += client_param.data.clone() / self.num_clients
        
        # save GPU memory
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        # 使用实际参与聚合的客户端数量
        actual_join_clients = len(self.uploaded_ids)
        
        # SCAFFOLD 的 global_c 更新应按“当前联邦训练问题中的有效客户端总数”缩放。
        # 正常训练时等于 self.num_clients；重训练/恢复排除目标客户端时，应排除这些客户端，
        # 否则控制变量会被错误稀释，训练轨迹明显偏离。
        excluded_count = len(getattr(self, "excluded_client_ids", []))
        actual_training_clients = self.num_clients - excluded_count
        if actual_training_clients <= 0:
            actual_training_clients = self.num_clients
        
        for cid in self.uploaded_ids:
            dy, dc = self.clients[cid].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / actual_join_clients * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / actual_training_clients
        self.global_model = global_model
        self.global_c = global_c

        # 同步buffer（尤其BN running stats）到全局模型，确保保存/重载一致
        # SCAFFOLD参数仍按delta_y更新；这里只聚合非参数buffer。
        if len(self.uploaded_ids) > 0:
            param_keys = {name for name, _ in self.global_model.named_parameters()}
            global_state = self.global_model.state_dict()
            client_states = [self.clients[cid].model.state_dict() for cid in self.uploaded_ids]

            for key, tensor in global_state.items():
                if key in param_keys:
                    continue
                if torch.is_floating_point(tensor):
                    agg = torch.zeros_like(tensor)
                    for w, c_state in zip(self.uploaded_weights, client_states):
                        agg += c_state[key] * w
                    global_state[key] = agg
                else:
                    global_state[key] = client_states[0][key].clone()

            self.global_model.load_state_dict(global_state, strict=False)

    @staticmethod
    def _clone_state_dict_to_cpu(state_dict):
        return {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
        }

    @staticmethod
    def _extract_buffer_state(model):
        param_keys = {name for name, _ in model.named_parameters()}
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key not in param_keys
        }

    @staticmethod
    def _extract_local_only_state(model):
        local_only_prefixes = tuple(getattr(model, "local_only_param_prefixes", ()))
        if not local_only_prefixes:
            return {}

        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if any(key.startswith(prefix) for prefix in local_only_prefixes)
        }

    def _capture_recovery_snapshot(self):
        snapshot = {
            "global_model_state": self._clone_state_dict_to_cpu(self.global_model.state_dict()),
            "client_buffer_states": {
                client.id: self._extract_buffer_state(client.model)
                for client in self.clients
            },
            "global_c": [param.detach().cpu().clone() for param in self.global_c],
            "client_c": {
                client.id: [param.detach().cpu().clone() for param in client.client_c]
                for client in self.clients
            },
        }
        client_local_param_states = {
            client.id: self._extract_local_only_state(client.model)
            for client in self.clients
            if self._extract_local_only_state(client.model)
        }
        if client_local_param_states:
            snapshot["client_local_param_states"] = client_local_param_states
        return snapshot

    def recovery_training(self, target_client_id, recovery_rounds=5, capture_snapshots=False):
        """
        恢复阶段训练：排除目标客户端，使用其他客户端数据进行训练
        
        Args:
            target_client_id: 要排除的目标客户端ID
            recovery_rounds: 恢复训练的轮数
            capture_snapshots: 是否保存每轮恢复后的模型快照
        """
        print(f"\n============= 开始恢复阶段训练 =============")
        print(f"排除目标客户端: {target_client_id}")
        print(f"恢复训练轮数: {recovery_rounds}")
        
        # 设置排除的客户端
        self.excluded_client_ids = [target_client_id]
        
        # 重置控制变量，使其与遗忘后的模型参数状态一致
        # 因为模型参数已经通过遗忘阶段移除了目标客户端的信息
        # 控制变量也应该重置，以保持一致性
        for param in self.global_c:
            param.data.zero_()
        for client in self.clients:
            for param in client.client_c:
                param.data.zero_()
        print(f"已重置global_c和所有客户端的client_c（恢复阶段，与遗忘后的模型参数状态一致）")
        
        # 保存原始参数
        original_global_rounds = self.global_rounds
        original_eval_gap = self.eval_gap
        
        # 设置恢复阶段的参数
        self.global_rounds = recovery_rounds
        self.eval_gap = 1  # 每轮都评估
        
        print(f"恢复阶段将进行 {recovery_rounds} 轮训练")
        print(f"参与训练的客户端: {[c.id for c in self.clients if c.id != target_client_id]}")
        
        # 执行恢复训练
        recovery_results = []
        for i in range(recovery_rounds):
            s_t = time.time()
            
            # 选择客户端（排除目标客户端）
            self.selected_clients = self.select_clients_with_exclusion()
            
            if len(self.selected_clients) == 0:
                print("警告：没有可用的客户端进行恢复训练")
                break
                
            print(f"\n--- 恢复阶段第 {i+1} 轮 ---")
            print(f"选中的客户端: {[c.id for c in self.selected_clients]}")
            
            # 发送模型
            self.send_models()
            
            # 客户端训练
            for client in self.selected_clients:
                client.train()
            
            # 接收模型
            self.receive_models()
            
            # 聚合参数
            self.aggregate_parameters()
            self.after_aggregation()
            
            # 评估模型
            if i % self.eval_gap == 0:
                print(f"\n评估恢复阶段第 {i+1} 轮模型")
                self.evaluate()
                round_result = {
                    'round': i + 1,
                    'test_acc': self.rs_test_acc[-1] if self.rs_test_acc else 0,
                    'test_auc': self.rs_test_auc[-1] if self.rs_test_auc else 0,
                    'train_loss': self.rs_train_loss[-1] if self.rs_train_loss else 0
                }
                if capture_snapshots:
                    round_result["snapshot"] = self._capture_recovery_snapshot()
                recovery_results.append(round_result)
            
            self.Budget.append(time.time() - s_t)
            print(f'恢复阶段第 {i+1} 轮时间成本: {self.Budget[-1]:.2f}s')
        
        # 恢复原始参数
        self.global_rounds = original_global_rounds
        self.eval_gap = original_eval_gap
        
        # 清除排除的客户端
        self.excluded_client_ids = []
        
        print(f"\n============= 恢复阶段训练完成 =============")
        print(f"恢复阶段结果:")
        for result in recovery_results:
            print(f"  第 {result['round']} 轮 - 测试准确率: {result['test_acc']:.4f}, 测试AUC: {result['test_auc']:.4f}")
        
        return recovery_results

    def retrain_without_client(self, target_client_id, retrain_rounds=None):
        """
        从头重训练：在整个训练阶段始终排除目标客户端

        Args:
            target_client_id: 需要排除的客户端 ID
            retrain_rounds: 重训练轮数（默认与 global_rounds 一致）
        """
        if retrain_rounds is None:
            retrain_rounds = self.global_rounds

        print(f"\n============= 开始重训练（排除客户端 {target_client_id}） =============")
        print(f"重训练轮数: {retrain_rounds}")

        # 设置排除的客户端
        self.excluded_client_ids = [target_client_id]

        # 清空历史指标，避免和之前训练混在一起
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        # 重置global_c和所有客户端的client_c（因为这是从头开始重训练）
        # 这是关键修复：在重训练时，SCAFFOLD的控制变量应该被重置
        for param in self.global_c:
            param.data.zero_()
        for client in self.clients:
            for param in client.client_c:
                param.data.zero_()
        print(f"已重置global_c和所有客户端的client_c（从头开始重训练）")

        # 备份原始设置
        original_global_rounds = self.global_rounds
        original_eval_gap = self.eval_gap

        # 使用新的轮数，并每轮评估一次
        self.global_rounds = retrain_rounds
        self.eval_gap = 1

        # 直接复用标准 train 的逻辑，但用带排除的采样函数
        for i in range(self.global_rounds+1):
            s_t = time.time()

            # 选择客户端（排除目标客户端）
            self.selected_clients = self.select_clients_with_exclusion()

            if len(self.selected_clients) == 0:
                print("警告：本轮没有可用的客户端，提前结束重训练")
                break

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Retrain Round number: {i}-------------")
                print("\nEvaluate global model (retrain)")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 恢复原始参数
        self.global_rounds = original_global_rounds
        self.eval_gap = original_eval_gap
        self.excluded_client_ids = []

        print("\n重训练完成。最佳准确率:")
        if len(self.rs_test_acc) > 0:
            print(max(self.rs_test_acc))
        else:
            print("暂无有效准确率记录。")

        return self.global_model

    def select_clients_with_exclusion(self):
        """选择客户端，排除指定的客户端"""
        # 获取可用的客户端（排除被排除的客户端）
        available_clients = [client for client in self.clients if client.id not in self.excluded_client_ids]
        
        if len(available_clients) == 0:
            print("警告：所有客户端都被排除，无法进行训练")
            return []
        
        # 计算实际参与训练的客户端数量
        if self.random_join_ratio:
            desired_num_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            desired_num_clients = self.num_join_clients
        
        # 确保不超过可用客户端数量
        actual_num_clients = min(desired_num_clients, len(available_clients))
        
        if actual_num_clients <= 0:
            print("警告：没有足够的客户端参与训练")
            return []
        
        selected_clients = list(np.random.choice(available_clients, actual_num_clients, replace=False))
        
        if len(self.excluded_client_ids) > 0:
            print(f"排除的客户端: {self.excluded_client_ids}，选择的客户端: {[c.id for c in selected_clients]}")
        
        return self._ensure_forced_client_selected(
            selected_clients,
            excluded_client_ids=self.excluded_client_ids,
        )

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.global_c)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
