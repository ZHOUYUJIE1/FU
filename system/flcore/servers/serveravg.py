import time
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.clients.clientfedau import clientFedAUAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        client_cls = clientFedAUAVG if getattr(args, "forget_strategy", "") == "fedau" else clientAVG
        self.set_clients(client_cls)

        # 支持在训练阶段排除某些客户端（用于重训练/恢复等）
        self.excluded_client_ids = []
        if getattr(args, "forget_strategy", "") == "fedau":
            self.force_include_client_id = getattr(args, "target_client_id", None)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


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
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

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

        # 更新当前参与训练的客户端数量
        self.current_num_join_clients = actual_num_clients

        selected_clients = list(np.random.choice(available_clients, actual_num_clients, replace=False))
        selected_clients = self._ensure_forced_client_selected(
            selected_clients,
            excluded_client_ids=self.excluded_client_ids,
        )

        if len(self.excluded_client_ids) > 0:
            print(f"排除的客户端: {self.excluded_client_ids}，选择的客户端: {[c.id for c in selected_clients]}")

        return selected_clients

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
        }
        client_local_param_states = {
            client.id: self._extract_local_only_state(client.model)
            for client in self.clients
            if self._extract_local_only_state(client.model)
        }
        if client_local_param_states:
            snapshot["client_local_param_states"] = client_local_param_states
        return snapshot

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

        # 备份原始设置
        original_global_rounds = self.global_rounds
        original_eval_gap = self.eval_gap

        # 使用新的轮数，并每轮评估一次
        self.global_rounds = retrain_rounds
        self.eval_gap = 1

        # 直接复用标准 train 的逻辑
        for i in range(self.global_rounds+1):
            s_t = time.time()
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
            self.after_aggregation()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 恢复设置 & 清理
        self.global_rounds = original_global_rounds
        self.eval_gap = original_eval_gap
        self.excluded_client_ids = []

        print("\n重训练完成。最佳准确率:")
        if len(self.rs_test_acc) > 0:
            print(max(self.rs_test_acc))
        else:
            print("暂无有效准确率记录。")

        return self.global_model

    def recovery_training(self, target_client_id, recovery_rounds=5, capture_snapshots=False):
        """
        恢复阶段训练：排除目标客户端，使用其他客户端数据进行训练
        
        Args:
            target_client_id: 要排除的目标客户端ID
            recovery_rounds: 恢复训练的轮数
            capture_snapshots: 是否保存每轮恢复后的模型快照
            
        Returns:
            list: 包含每轮训练结果的列表，每个元素包含 round, test_acc, test_auc, train_loss
        """
        print(f"\n============= 开始恢复阶段训练 =============")
        print(f"排除目标客户端: {target_client_id}")
        print(f"恢复训练轮数: {recovery_rounds}")
        
        # 设置排除的客户端
        self.excluded_client_ids = [target_client_id]
        
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
        self.excluded_client_ids = []
        
        print(f"\n============= 恢复阶段训练完成 =============")
        print(f"恢复阶段结果:")
        for result in recovery_results:
            print(f"  第 {result['round']} 轮 - 测试准确率: {result['test_acc']:.4f}, 测试AUC: {result['test_auc']:.4f}")
        
        return recovery_results
