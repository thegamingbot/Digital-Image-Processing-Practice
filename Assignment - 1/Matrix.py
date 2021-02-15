class Matrix:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.matrix = []

    def __repr__(self):
        out = ""
        for i in self.matrix:
            for j in i:
                out = out + str(j) + "\t"
            out += "\n"
        return out

    def matrixSum(self):
        num = 0
        for i in self.matrix:
            for j in i:
                num += j
        return num

    def matrixMax(self):
        num = -100000
        for i in self.matrix:
            num1 = max(i)
            if num1 > num:
                num = num1
        return num

    def matrixMean(self):
        return self.matrixSum() / (self.m * self.n)

    def matrixMedian(self):
        matrixList = self.__matrixListGen()
        mid = len(matrixList) // 2
        return (matrixList[mid] + matrixList[~mid]) / 2

    def matrixMode(self):
        matrixList = self.__matrixListGen()
        return max(set(matrixList), key=matrixList.count)

    def matrixStdDeviation(self):
        variance = 0
        mean = self.matrixMean()
        for i in self.matrix:
            for j in i:
                variance += ((j - mean) ** 2)
        return (variance / (self.m * self.n)) ** 0.5

    def matrixFreqDistribution(self):
        freq = {}
        for i in self.matrix:
            for j in i:
                if freq.get(j):
                    freq[j] += 1
                else:
                    freq[j] = 1
        return freq

    def __matrixListGen(self):
        matrixList = []
        for i in self.matrix:
            matrixList += i
        return matrixList
