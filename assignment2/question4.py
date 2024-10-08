import scipy.stats as stats

class_1_test = [0.4015, 0.3995, 0.3991, 0.4003, 0.3985, 0.3998, 0.3997]
class_2_test = [0.3247, 0.3360, 0.2974, 0.2554, 0.3139, 0.2627, 0.3802]
class_3_test = [0.4443, 0.5505, 0.6469, 0.5632, 0.7687, 0.0524, 0.7586]

# Calculate PDF at x = 0 for standard normal distribution
def finding_fx(x):

    classes= {'class 1': [0.4, 0.01], 'class 2': [0.32, 0.05], 'class 3': [0.55, 0.02]}

    maximum = 0
    class_given = 'none'
    for key, value in classes.items():
        pdf_value = stats.norm.pdf(x, loc=value[0], scale=value[1])
        # print(pdf_value)
        f_y = 1/3
        result = pdf_value * f_y
        if result > maximum:
            class_given = key
            maximum = result


    return result, class_given

print("Items from class 1")
for item in class_1_test:
    lowest_difference, list_name = finding_fx(item)
    print(f'Item {item} is in {list_name}. The probability is {lowest_difference}')

print("Items from class 2")
for item in class_2_test:
    lowest_difference, list_name = finding_fx(item)
    print(f'Item {item} is in {list_name}. The probability is {lowest_difference}')

print("Items from class 3")
for item in class_3_test:
    lowest_difference, list_name = finding_fx(item)
    print(f'Item {item} is in {list_name}. The probability is {lowest_difference}')