universe = vanilla
executable = wrapper_postprocessing_ticl.sh
arguments = $(Process) zll_0pu_v0.txt
getenv = TRUE
output = logs/post_$(Process).out
error  = logs/post_$(Process).err
log    = logs/post_build.log
transfer_input_files = postprocessing_ticl_ttbar_nopu.py, zll_0pu_v0.txt
transfer_output_files = ""
notification = never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
+JobFlavour = "tomorrow"
Queue 1600
