## Config OpenCV:

#### Project -> Property -> C/C++ -> General -> Additional Include Directories:
C:\opencv-4.1.0\opencv-4.1.0\build\install\include

#### Project -> Property -> Linker -> General -> Additional Library Directories:
C:\opencv-4.1.0\opencv-4.1.0\build\install\x64\vc15\lib

#### Project -> Property -> Linker -> Input -> Additional Dependencies:
Enter lib names. (debug / release)


## Config Dlib:

#### Project -> Property -> VC++ Directories -> Include Directories:
C:\dlib\source

#### Project -> Property -> VC++ Directories -> Library Directories:
C:\dlib\build\dlib\Release

#### Project -> Property -> Linker -> Input -> Additional Dependencies:
Enter lib names. (debug / release)

#### Project -> Property -> C/C++ -> Preprocessor -> Preprocessor Definitions:
Enter: DLIB_JPEG_SUPPORT/ DLIB_USE_CUDA /... OR add dlib/all/source.cpp into project
