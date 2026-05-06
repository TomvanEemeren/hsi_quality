# import matplotlib.pyplot as plt
# from hsi_quality.metrics import calculate_mean_ssim, calculate_mvssim, calculate_q_lambda

# def plot_mean_ssim(dataset):
    # angles = dataset["off_nadir"]

    # scores = {}
    # for idx, data in enumerate(resampled_data):
    #     angle = angles[idx]
    #     score = calculate_mean_ssim(resampled_data[0], resampled_data[idx])
    #     scores[angle] = score

    # plt.figure(figsize=(10, 5))
    # plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    # plt.xlabel("Off-nadir angle (degrees)")
    # plt.ylabel("Mean SSIM score")
    # plt.grid()
    # plt.show()