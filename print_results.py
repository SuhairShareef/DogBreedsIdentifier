#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#
# PROGRAMMER:   Suhair Shareef
# DATE CREATED: June 6, 2023.
# REVISED DATE: June 6, 2023.
# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results
#          dictionary (results_dic).
#         This function inputs:
#            -The results dictionary as results_dic within print_results
#             function and results for the function call within main.
#            -The results statistics dictionary as results_stats_dic within
#             print_results function and results_stats for the function call within main.
#            -The CNN model architecture as model wihtin print_results function
#             and in_arg.arch for the function call within main.
#            -Prints Incorrectly Classified Dogs as print_incorrect_dogs within
#             print_results function and set as either boolean value True or
#             False in the function call within main (defaults to False)
#            -Prints Incorrectly Classified Breeds as print_incorrect_breed within
#             print_results function and set as either boolean value True or
#             False in the function call within main (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.


def print_results(
    results_dic,
    results_stats_dic,
    model,
    print_incorrect_dogs=False,
    print_incorrect_breed=False,
):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    print(
        f"""
            -----------   Dog Breed Identifier using {model} ML model   -----------\n\n\n\n
                                           Results                                  \n\n
            Counter:\n\n
            Number of images: {results_stats_dic["n_images"]}\n
            Number of Dog Images: {results_stats_dic["n_dogs_img"]}\n
            Number of "Not-a" Dog Images: {results_stats_dic["n_notdogs_img"]}\n\n
            
            Percentages:\n\n
            % Correct Dogs: {results_stats_dic["pct_correct_dogs"]}\n
            % Correct Breed: {results_stats_dic["pct_correct_breed"]}\n
            % Correct "Not-a" Dog: {results_stats_dic["pct_correct_notdogs"]}\n
            % Match: {results_stats_dic["pct_match"]}\n\n
            
        """
    )
    misclassified_images_count = (
        results_stats_dic["n_correct_dogs"] + results_stats_dic["n_correct_notdogs"]
    )

    if misclassified_images_count != results_stats_dic["n_images"]:
        print(
            f"There are {results_stats_dic['n_images'] - misclassified_images_count} "
            "image(s) that has been misclassified!\n"
        )

    if (
        print_incorrect_dogs
        and misclassified_images_count != results_stats_dic["n_images"]
    ):
        print("The following pets has been misclassified:\n")
        for file in results_dic:
            if sum(results_dic[file][3:]) == 1:
                print(
                    f"Image name: '{file}', "
                    f"Pet label: '{results_dic[file][0]}', "
                    f"Classification label: '{results_dic[file][1]}'"
                )

    if (
        print_incorrect_breed
        and results_stats_dic["n_correct_dogs"] != results_stats_dic["n_correct_breed"]
    ):
        print("The following dogs has been misclassified in breed:\n")
        for file in results_dic:
            if sum(results_dic[file][3:]) == 2 and results_dic[file][2] == 0:
                print(
                    f"Image name: '{file}', "
                    f"Pet label: '{results_dic[file][0]}', "
                    f"Classification label: '{results_dic[file][1]}'"
                )
