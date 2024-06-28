import argparse
import subprocess
import trimesh
import os

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

        model = trimesh.load(f'../data/3d_models/{save_path}_mesh.obj')
        model.export(f'../data/3d_models/{save_path}_mesh.obj')
        os.remove(f'../data/3d_models/{save_path}_mesh.mtl')
        os.remove(f'../data/3d_models/{save_path}_mesh_albedo.png')
        with open(f'../data/3d_models/{save_path}_mesh.obj', 'r') as file:
            file_contents = file.read()

        # Replace the old string with the new string
        updated_contents = file_contents.replace('material0.mtl', save_path+'_mesh.mtl')
        updated_contents = updated_contents.replace('material0', save_path+'_mesh')

        # Write the modified contents back to the file
        with open(f'../data/3d_models/{save_path}_mesh.obj', 'w') as file:
            file.write(updated_contents)

        with open(f'../data/3d_models/material0.mtl', 'r') as file:
            file_contents = file.read()

        # Replace the old string with the new string
        updated_contents = file_contents.replace('material0.png', save_path+'_mesh_albedo.png')
        updated_contents = updated_contents.replace('material0', save_path+'_mesh')

        # Write the modified contents back to the file
        with open(f'../data/3d_models/material0.mtl', 'w') as file:
            file.write(updated_contents)

        os.rename(f'../data/3d_models/material0.mtl',f'../data/3d_models/{save_path}_mesh.mtl')
        os.rename(f'../data/3d_models/material0.png',f'../data/3d_models/{save_path}_mesh_albedo.png')
        os.remove(f'../data/3d_models/{save_path}.obj')
        os.remove(f'../data/3d_models/{save_path}.mtl')
        os.remove(f'../data/3d_models/{save_path}_albedo.png')
        #os.remove(f'../data/3d_models/{save_path}_albedo.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a series of Python scripts with specified arguments.')
    parser.add_argument('name', type=str, help='The name of the image file (e.g., input/name.jpg)')
    parser.add_argument('--size', type=int, default=512, help='The size argument for the first command (default: 512)')

    args = parser.parse_args()

    runner = ImgTo3dPipeline()
    runner(args.name, args.size)
