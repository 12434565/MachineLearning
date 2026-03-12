def predict_given_tree(temp_2, temp_1, average, friend):
    # Only temp_1 and average are used in this tree (for this path)
    if temp_1 <= 59.5:
        if average <= 46.8:
            if temp_1 <= 44.5:
                return 41.0
            else:
                return 45.0
        else:
            # (not needed for your sample)
            return 58.2  # placeholder
    else:
        # (not needed for your sample)
        return 73.0

pred = predict_given_tree(temp_2=39, temp_1=35, average=44, friend=30)
print("Prediction (given tree):", pred)
print("Variables used:", ["temp_1", "average"])