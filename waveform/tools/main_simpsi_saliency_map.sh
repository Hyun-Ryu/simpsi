#!/bin/bash
PRIOR="slc"

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description jitter --aug_list jitter --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description jitter --aug_list jitter --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description jitter --aug_list jitter --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description magwarp --aug_list magwarp --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description magwarp --aug_list magwarp --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description magwarp --aug_list magwarp --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description dropout --aug_list dropout --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description dropout --aug_list dropout --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description dropout --aug_list dropout --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description timewarp --aug_list timewarp --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description timewarp --aug_list timewarp --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description timewarp --aug_list timewarp --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description permute --aug_list permute --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description permute --aug_list permute --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description permute --aug_list permute --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description freqdropout --aug_list freqdropout --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description freqdropout --aug_list freqdropout --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description freqdropout --aug_list freqdropout --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2

python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description scale_shift_jitter --aug_list scale shift jitter --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 0
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description scale_shift_jitter --aug_list scale shift jitter --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 1
python -m evaluations.main_nonctr --experiment_description waveform_${PRIOR} --run_description scale_shift_jitter --aug_list scale shift jitter --prior ${PRIOR} --equalizer conv --eq_kernel_size 25 --seed 2
