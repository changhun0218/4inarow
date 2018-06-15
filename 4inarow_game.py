import numpy as np

class game:
    whose_turn = 1
    vectors = np.array([[1, 0], [1, 1], [0, 1], [1, -1]])
    def __init__(self):
        self.board = np.zeros((7, 9,))
        
    def board_update(self, i):
        num_non_zero = np.count_nonzero(self.board[:, i + 1])
        x, y = 5 - num_non_zero, i + 1
        if num_non_zero == 6: # over
            return

        self.board[x, y] = self.whose_turn

        for vector in self.vectors:
            count = 0
            pos = np.array([x, y])
            print("pos:", pos)
            for sign in [-1, 1]:
                temp = np.copy(pos)
                while count < 10:
                    temp += vector * sign
                    print(sign, temp)
                    if self.board[temp[0], temp[1]] == self.whose_turn:
                        count += 1
                    else :
                        break
            if count >= 3:
                print(self.board)
                print("Winner is :", self.whose_turn)
                raise
        self.whose_turn *= -1
        
    def make_a_move(self):
        print(self.board)
        i = np.random.randint(6)
        #        i = int(input("where ?"))
        self.board_update(i)

        
def main():
    play = game()
    i = 0
    while i < 1:
        play.make_a_move()
        
    
    
if __name__ == "__main__":
    main()
