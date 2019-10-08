# pyFastcamTools
## Python 2 library containing fast camera analysis tools

This repository contains a suite of tools developed in python primarily for the analysis of fast camera data.

Some of the tools may be more widely applicable.

If you add any features to any of the tools, or which to add your own tool, please include in the repository.

### Currently contained in the repo
**_FILE_**  | **_DESCRIPTION_**  
---|---
**backgroundSubtractor.py**		|	*A class to subtract the background from a given image and return the foreground.* 
**backgroundSubtractor_example.py** 	|	*Example script running a background subtractor on an .avi video file using openCV*
**cv2_opticalFlow.py**			| 	*A script using the openCV dense optical flow tool to display the optical flow of a video as a quiver plot*
**equilibrium.py**			|	*A class to store load in and store useful information about a plasma equilibrium and abstract the user from the data source.* 
**fieldlineTracer.py**			| 	*A class to create a fieldline tracer which takes an equilibrium and returns the trajectory of a field line from a given starting position.*
**fieldlineTracer_example.py**		| 	*Example script using the fieldline tracer class*
**frameHistory.py**			|	*An object to store a movie as a history of frames which is accessible by indexing but also contains some useful functions*
**geqdsk.py**				|	*A useful class to read in data from efit gfiles*
**PlayMovie.py**			| 	*A very simple script to read in a play a .avi movie file using openCV*
**readMovie.py**			|	*A function to read in frames from a movie file (including MAST .ipx files) and store the result as a frameHistory instance*
**pyAutoGit.py**         	 	|	*Callable class to automatically commit any changes to the repo at the end of analysis scripts*
**filament_tracker/Elzar.py**		|	*A class to analyse data from Fast cameras. Contains a GUI writted using matplotlib widgets*
**filament_tracker/utils.py**		| 	*Some utilities used in Elzar*


