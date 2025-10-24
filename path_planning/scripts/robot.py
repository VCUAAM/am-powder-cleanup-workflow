from utils import sin, cos

class Robot(object):

    def __init__(self, matrix, start_position, start_direction, robot_size):
        self.matrix = matrix
        self.current_position = {'x': start_position['x'], 'y': start_position['y']}
        self.current_direction = start_direction
        self.__visited_position = {str(start_position['x']) + '_' + str(start_position['y']): 1}
        self.move_count = 0
        self.turn_count = 0
        self.loggable = False
        self.path = []
        # self.path.append((self.current_position['x'], self.current_position['y']))
        self.robot_size = robot_size    

    def turn_left(self):
        """turn 90 degree counter-clockwise"""
        self.current_direction = (self.current_direction + 1) % 4
        self.turn_count += 1
        return self

    def turn_right(self):
        """turn 90 degree clockwise"""
        self.current_direction = (self.current_direction + 3) % 4
        self.turn_count += 1
        return self

    # def move(self):
    #     """Move ahead with offset step size"""
    #     next_pos_x = self.current_position['x'] + self.robot_size * cos(self.current_direction)
    #     next_pos_y = self.current_position['y'] - self.robot_size * sin(self.current_direction)

    #     if not self.__can_move(next_pos_x, next_pos_y):
    #         return False

    #     self.move_count += 1
    #     self.current_position['x'] = next_pos_x
    #     self.current_position['y'] = next_pos_y
    #     self.__visited_position[str(next_pos_x) + "_" + str(next_pos_y)] = 1
    #     self.path.append((next_pos_x, next_pos_y))  

    #     if self.loggable:
    #         self.log()
    #     return True

    def move(self):
        """Move ahead with offset step size"""
        next_pos_x = self.current_position['x'] + self.robot_size * cos(self.current_direction)
        next_pos_y = self.current_position['y'] - self.robot_size * sin(self.current_direction)

        if not self.__can_move(next_pos_x, next_pos_y):
            return False

        self.move_count += 1

        # Compute movement direction
        dx = next_pos_x - self.current_position['x']
        dy = next_pos_y - self.current_position['y']

        # Append position only if direction changes or path is empty
        if not self.path or (dx, dy) != getattr(self, "last_direction", (None, None)):
            self.path.append((self.current_position['x'], self.current_position['y']))
            # print(self.current_position)
            self.last_direction = (dx, dy)  # Store last move direction

        self.current_position['x'] = next_pos_x
        self.current_position['y'] = next_pos_y
        self.__visited_position[f"{next_pos_x}_{next_pos_y}"] = 1

        if self.loggable:
            self.log()
        return True


    def __can_move(self, next_pos_x, next_pos_y):
        """Check if the robot can move while considering its size (offset)"""
        for i in range(self.robot_size):
            for j in range(self.robot_size):
                check_x = next_pos_x + i
                check_y = next_pos_y + j
                if (
                    check_x < 0 or check_y < 0 or
                    check_y >= len(self.matrix) or check_x >= len(self.matrix[0]) or
                    self.matrix[check_y][check_x] == 1
                ):
                    return False
        return True

    def get_path(self):
        return self.path
    
    def log(self):
        for i in range(len(self.matrix)):
            text = ""
            for j in range(len(self.matrix[i])):
                if i == self.current_position['y'] and j == self.current_position['x']:
                    if self.current_direction == 0:
                        text += '>'
                    elif self.current_direction == 1:
                        text += '^'
                    elif self.current_direction == 2:
                        text += '<'
                    else:
                        text += 'v'
                elif self.__visited_position.get(str(j) + "_" + str(i), None) == 1:
                    text += '*'
                elif self.matrix[i][j] == 0:
                    text += '.'
                else:
                    text += '|'
        #     print(text)
        # print('')