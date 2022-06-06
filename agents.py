import numpy as np

def main():
    print('Main')

class agent:
    def __init__(self, dynamics, personal_cost, initial_state, current_action, goal_state):
        self.dynamics = dynamics
        self.personal_cost = personal_cost
        self.initial_state = initial_state
        self.current_action = current_action
        self.goal_state = goal_state
        self.current_state = initial_state

    def execute_action(self):
        # This is where the simulation will run and the agent will execute the action
        return None


if __name__ == '__main__':
    main()
