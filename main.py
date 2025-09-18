from image_status_updater.image_status_updater_v2_0 import main as update_image_statuses
from image_trimmer.image_trimmer_v1_0 import main as trim_images
from pomelo_extractor.pomelo_extractor_v2_12 import main as extract_pomelos

def load_pomelo_extractor():
    print('Coming soon...')

def load_trimmer():
    trim_images()
    print('Done')

choices = {
    "Load pomelo extractor": load_pomelo_extractor,
    "Load trimmer": load_trimmer
}
default_choice_number = 2

def main():
    choice_keys = list(choices.keys())
    choice_range = range(len(choice_keys))
    print()
    for index, choice in enumerate(choices):
        print(f"({index+1}) {choice}")
    print()
    if default_choice_number == None:
        choice_index = int(input("Enter number: ")) - 1
    else:
        print(f"(Default) Running choice #{default_choice_number}")
        choice_index = default_choice_number - 1
    print()
    if choice_index not in choice_range:
        print("\033[31mInvalid choice\033[0m")
    else:
        choices[choice_keys[choice_index]]()

if __name__ == "__main__":
    main()
