def load_pomelo_extractor():
    print('nice')

def load_trimmer_and_csv_updater():
    print('ok')

choices = {
    "Load pomelo extractor": load_pomelo_extractor,
    "Load trimmer and CSV updater": load_trimmer_and_csv_updater
}

def main():
    choice_keys = list(choices.keys())
    choice_range = range(len(choice_keys))
    print()
    for index, choice in enumerate(choices):
        print(f"({index+1}) {choice}")
    choice_index = int(input("\nEnter number: ")) - 1
    print()
    if choice_index not in choice_range:
        print("\033[31mInvalid choice\033[0m")
    else:
        choices[choice_keys[choice_index]]()

if __name__ == "__main__":
    main()
