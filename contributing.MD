* test
  * **test**
  * ***test***

# heading
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

Getting Set Up
===============

* Download the latest version of python 3:
  * Should be 3.10 or later
  * https://www.python.org/downloads/
  
* Install Ubuntu through WSL on Windows:
  * install WSL:
    * If you experience any issues while following the tutorial linked below, try splitting out the copy/pasted code and do it line by line as a first step.
    * https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#2-install-wsl
  * install Ubuntu:
    * version 22.04
    * https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#3-download-ubuntu

* Install Ros 2 Rolling Ridely:
  * https://docs.ros.org/en/rolling/Installation/Ubuntu-Install-Debians.html
    
* Install Virtual Studio Code (VS Code):
  * https://code.visualstudio.com/download
  * Additional extensions can be downloaded either through the links provided below or by searching the name of the extension in the extensions menu in VS Code
  * Extensions that will be needed:
    * PowerShell
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode.PowerShell
    * Pylance
      * https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance
    * C/C++
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools
    * C/C++ Extension Pack
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack
    * C/C++ Themes
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-themes
    * CMake
      * https://marketplace.visualstudio.com/items?itemName=twxs.cmake
    * CMake Tools
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools
    * IntelliCode
      * https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode
    * IntelliCode API Usage Examples
      * https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.intellicode-api-usage-examples
    * isort
      * https://marketplace.visualstudio.com/items?itemName=ms-python.isort
    * Jupyter
      * https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
    * Jupyter Cell Tags
      * https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-cell-tags
    * Jupyter Keymap
      * https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-keymap
    * Jupyter Notebook Renderers
      * https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers
    * Jupyter Slide Show
      * https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-slideshow
    * MagicPython
      * https://marketplace.visualstudio.com/items?itemName=magicstack.MagicPython
    * Python
      * https://marketplace.visualstudio.com/items?itemName=ms-python.python
    * Remote - SSH
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh
    * Remote - SSH: Editing Configuration Files
      * https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh-edit
    * ROS2
      * https://marketplace.visualstudio.com/items?itemName=nonanonno.vscode-ros2
    * SonarLint
      * https://marketplace.visualstudio.com/items?itemName=SonarSource.sonarlint-vscode
      
## 🚀 Getting Started
```bash
git clone https://github.com/VCUAAM/vcu_am_post_processing.git
cd vcu_am_post_processing
pip install -r requirements.txt
```

Example:
```bash
python master_script.py
```