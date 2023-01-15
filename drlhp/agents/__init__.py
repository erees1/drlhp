from agents.dqn_agent import DQNAgent, DQNFixedTarget


def get_agent_class(type):
    if type == "dqn":
        return DQNAgent
    elif type == "dqn_fixed_target":
        return DQNFixedTarget
