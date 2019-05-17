import os
import locale
import logging
import time
import datetime
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer

class PolicyLearner:
    def __init__(self, stock_code, chart_data, training_data=None,
                min_trading_unit=1, max_trading_unit=2,
                delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code # 종목코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data) # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment,
                        min_trading_unit=min_trading_unit,
                        max_trading_unit=max_trading_unit,
                        delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer() # 가시화 모듈
    
    def reset(self):
        self.sample = None
        self.training_data_idx = -1
    
    def fit(self,
            num_epoches=1000, max_memory=60, balance=1000000,
            discount_factor=0, start_epsilon=.5, learning=True):
        logger.info(f'LR: {self.policy_network.lr}, DF: {discount_factor}, '
                    f'TU: [{self.agent.min_trading_unit}, {self.agent.max_trading_unit}], '
                    f'DRT: {self.agent.delayed_reward_threshold}')
        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, f'epoch_summary/{self.stock_code}/epoch_summary_{settings.timestr}')
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)
        
        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            # 에포크 관련 정보 초기화
            loss = 0.
            itr_cnt = 0
            win_cnt = 0
            exploration_cnt = 0
            batch_size = 0
            pos_learning_cnt = 0
            neg_learning_cnt = 0

            # 메모리 초기화
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []