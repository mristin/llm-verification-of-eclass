input1: str = input("Definition 1: ")
input2: str = input("Definition 2: ")

if input1 == input2:
    print("EQUAL")
else:
    print("NOT EQUAL")
    print("Equal until: ")
    for i in range(len(input1)):
        if input1[i] != input2[i]:
            print(f"letter {i}:\n-->{input1[i:]}")
            print(f"\n-->{input2[i:]}")
            break
