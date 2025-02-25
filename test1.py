import numpy as np
import matplotlib.pyplot as plt

class Q1:
    def __init__(self):
        self.averages = np.array([33,34,40,51,60,69,75,74,67,56,47,38])
        self.points = [(16 + 31 * i, self.averages[i]) for i in range(len(self.averages))]
    def part1(self):
        matrix = np.array([[np.pow(i,k) for k in range(4)] for i, _ in self.points])
        transpose = np.linalg.matrix_transpose(matrix)
        matrix = np.linalg.matmul(transpose, matrix)
        self.averages = np.linalg.matmul(transpose, self.averages)
        self.coefficients = np.linalg.matmul(np.linalg.inv(matrix), self.averages)
        self.f = lambda x: sum(self.coefficients[i] * np.pow(x,i) for i in range(len(self.coefficients)))
        x = np.linspace(16 + 31 * 0, 16 + 31 * 12, 1000)
        plt.plot(x, [self.f(i) for i in x], label = f'P3(x)')
        plt.legend()
        for a,b in self.points:
            plt.plot(a,b, 'ro')
        print(f'Coefficients: {','.join([str(c) for c in self.coefficients])}')
        plt.show()
        
    def part2(self):
        return(f'Average Temperature on June 4: {self.f(4 + 31 * 6)}\nAverage Temperature on December 25: {self.f(25 + 31 * 12)}')
    
    def part3(self):
        self.coefficients[0] -= 64.89
        self.f = lambda x: sum(self.coefficients[i] * np.pow(x,i) for i in range(len(self.coefficients)))
        self.derivative = lambda x: sum(i * self.coefficients[i] * np.pow(x,i - 1) for i in range(1,len(self.coefficients)))
        tol = 0.5 * (10 ** -4)
        x = 0
        while(abs(self.f(x)) >= tol):
            x -= (self.f(x) / self.derivative(x))
        return f'Month:{int(x) // 31}, Day:{int(x) % 31}'
    
if __name__ == '__main__':
    Q1 = Q1()
    Q1.part1()
    print(Q1.part2())
    print(Q1.part3())