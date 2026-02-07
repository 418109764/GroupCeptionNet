import sys
import warnings
import argparse
from functools import partial
from loaders import chiliseeds_loader
from results import save_results
from utils.enums import Tasks
from classifiers import TestTaskClassifier

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        type=str,
        help="Select the desired optimization technique [sgd/adam].",
        default="sgd",
    )
    parser.add_argument("--batch_size", type=int, help="Set images batch size", default=18)
    parser.add_argument(
        "--lr", type=float, help="Set the learning rate for the optimizer", default=0.0001
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Set L2 parameter norm penalty", default=5e-5
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        help="Select the data augmentation technique [standard/mixup/bc+]",
        default="standard",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select CNN architecture [resnet34/resnet50/resnet101/alexnet/googlenet/vgg16/mobilenet_v2]",
        default="resnet50",
    )
    parser.add_argument("--epochs", type=int, help="Set the number of epochs.", default=40)
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Defines whether or not to use a pre-trained model.",
        default=True,
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path of the dataset csv file.",
        default="pepper_seed_KCI.csv",  # pepper_seed.csv
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="The data is changed based on the selected fold [1-5].",
        default=1,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes in the dataset.",
        default=2,
    )
    parser.add_argument(
        "--pca_factor",
        type=int,
        help="The PCA factor to reduce the image depth.",
        default=1,
    )
    parser.add_argument(
        "--target",
        type=int,
        help="The target to predict.[0-5]",
        default=2,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment.",
        default="experiment",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to the results folder.",
        default="results",
    )
    parser.add_argument("--train", help="Run in training mode.", action="store_true")
    parser.add_argument("--test", help="Run in test mode.", action="store_true")
    parser.add_argument(
        "--dataset",
        help="Select the dataset to use. Options: process_dataset, process_dataset_without_cutout",
        type=str,
        default="process_dataset",
    )
    parser.add_argument(
        "--model_task",
        help=(
            "Select the model task according to the dataset. "
            "Dataset: (0) process_dataset"
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to use in the data loader. Default is 9.",
        type=int,
        default=2,
    )

    # Parse the arguments
    options = parser.parse_args()

    if options.dataset == "process_dataset":
        options.model_task = Tasks(options.model_task)
    else:
        options.model_task = Tasks(0)

    # Validate the arguments
    assert (
        options.train or options.test
    ), "You must specify wheter you want to train or test a model."

    assert options.dataset in [
        "process_dataset",
        "process_dataset_without_cutout",
    ], "You must specify a valid dataset."

    # Initialize the classifier
    Clf = TestTaskClassifier(
        images_dir=f"{options.dataset}",
        csv_file=options.csv_file,
        fold=options.fold,
        num_classes=options.num_classes,
        model_task=options.model_task,
        batch_size=options.batch_size,
        epochs=options.epochs,
        model=options.model,
        pretrained=options.pretrained,
        optimizer=options.optimizer,
        lr=options.lr,
        weight_decay=options.weight_decay,
        data_augmentation=options.data_augmentation,
        results_path=options.results_path,
        experiment_name=options.experiment_name,
    )

    loader = partial(
        chiliseeds_loader,
        csv_file=options.csv_file,
        fold=options.fold,
        model_task=options.model_task,
        num_workers=options.num_workers,
        pca_factor=options.pca_factor,
        target=options.target,
        model=options.model,
    )

    # Run the classifier
    if options.train:
        Clf.run_training(loader)

    elif options.test:
        out = Clf.run_test(loader)
        save_results(out, options.model, options.results_path, options.experiment_name)

    else:
        raise ValueError("You must specify wheter you want to train or test a model.")