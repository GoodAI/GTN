import os

params = [
    'epochs=5000 inner_loop_steps=20 batch_size=32 teacher_input_type=learned',
    'epochs=5000 inner_loop_steps=20 batch_size=32 teacher_input_type=random_fixed',
    'epochs=5000 inner_loop_steps=20 batch_size=32 teacher_input_type=random',

    'epochs=5000 inner_loop_steps=10 batch_size=32 teacher_input_type=learned',
    'epochs=5000 inner_loop_steps=10 batch_size=32 teacher_input_type=random_fixed',
    'epochs=5000 inner_loop_steps=10 batch_size=32 teacher_input_type=random',

    'epochs=5000 inner_loop_steps=10 batch_size=16 teacher_input_type=learned',
    'epochs=5000 inner_loop_steps=10 batch_size=16 teacher_input_type=random_fixed',
    'epochs=5000 inner_loop_steps=10 batch_size=16 teacher_input_type=random',
]
common = ' '
for param in params:
    merged_param = f'{common} {param}'
    print(f'Running with params {merged_param}')
    os.system(f'python mnist_experiment.py with {merged_param}')
