import argparse
import subprocess


class ImgTo3dPipeline:
    def __call__(self, name, size):
        self.run_commands(name, size)

    def run_commands(self, name, size):
        # Run the first command
        subprocess.run(['python', 'process.py','../' + name, '--size', str(size)])

        # Derive the inputs for the next commands
        name_rgba = '../' + name.replace('.png', '_rgba.png')
        # Need to add folder input
        save_path = name.replace('data/images/', '').replace('.png', '')
        save_path = save_path.split('/')[-1]
        save_path = save_path.split('\\')[-1]
        # Run the second command
        subprocess.run(
            ['python', 'main.py', '--config', 'configs/image_sai.yaml', f'input={name_rgba}', f'save_path={save_path}'])

        # Run the third command
        subprocess.run(
            ['python', 'main2.py', '--config', 'configs/image_sai.yaml', f'input={name_rgba}',
             f'save_path={save_path}'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a series of Python scripts with specified arguments.')
    parser.add_argument('name', type=str, help='The name of the image file (e.g., input/name.jpg)')
    parser.add_argument('--size', type=int, default=512, help='The size argument for the first command (default: 512)')

    args = parser.parse_args()

    runner = ImgTo3dPipeline()
    runner(args.name, args.size)
