from Matrix import Matrix

if __name__ == "__main__":
    matrix = Matrix(5, 5)

    for i in range(matrix.m):
        a = []
        for j in range(matrix.n):
            a.append(
                int(input("\x1b[0;34;40m" + f"Enter input at index [{i}][{j}]: " + "\x1b[0m")))
        matrix.matrix.append(a)
    print("\x1b[0;32;40m" + "\nInput matrix: " + "\x1b[0m")
    print(matrix.__repr__())

    functions = {
        1: ["Matrix Sum", matrix.matrixSum],
        2: ["Matrix Maximum", matrix.matrixMax],
        3: ["Matrix Mean", matrix.matrixMean],
        4: ["Matrix Median", matrix.matrixMedian],
        5: ["Matrix Mode", matrix.matrixMode],
        6: ["Matrix Standard Deviation", matrix.matrixStdDeviation],
        7: ["Matrix Frequency Distribution", matrix.matrixFreqDistribution]
    }

    while True:
        choice = int(input(
            "\x1b[0;33;40m" + "\n1. Sum\n2. Maximum\n3. Mean\n4. Median\n5. Mode\n6. Standard Deviation\n7. Frequency Distribution\n8. Exit\nEnter the number corresponding to the input function: " + "\x1b[0m"))
        if choice >= 8:
            break
        print(functions[choice][0] + ": " + str(functions[choice][1]()))
