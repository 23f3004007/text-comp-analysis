# In-Depth Analysis of Text Compression Algorithms
## Overview
In the realm of operating systems and data management, efficient storage and transmission of data are paramount. This project provides an in-depth comparative analysis of five foundational lossless text compression algorithms. It is implemented as a hands-on, interactive web application that allows users to upload text files and receive a real-time performance breakdown for each algorithm.

The core objective is to move beyond theoretical understanding and provide practical, quantitative insights into the trade-offs between compression efficiency (ratio) and computational cost (speed). This tool serves as both a powerful educational resource and a functional utility for developers and students in the field of computer science.

This project was developed as a partial fulfillment for the Operating Systems course (BCSE303L) at Vellore Institute of Technology, Chennai.

## Key Features
### Comprehensive Algorithm Analysis: The application implements and evaluates five distinct algorithms:
- Run-Length Encoding (RLE)
- Huffman Coding
- Lempel-Ziv-Welch (LZW)
- DEFLATE (the algorithm behind ZIP and GZIP)
- Arithmetic Coding

### Quantitative Performance Metrics: For each algorithm, the tool calculates and displays three critical metrics:
- Compression Ratio: The ultimate measure of space-saving effectiveness.
- Compression Time: The time taken to compress the data, indicating computational efficiency.
- Decompression Time: The speed of data restoration, crucial for read-heavy applications.
- Interactive Web Interface: A user-friendly UI allows for:
  Easy uploading of any .txt file.
  A clear, tabular presentation of the results.
  The ability to preview the decompressed text to ensure data integrity.
  Direct download links for both the compressed binary file and the restored text file.

## Application Screenshot
The main interface provides a clean and intuitive summary of the analysis, allowing for easy comparison between the different methods.
<img width="790" height="369" alt="image" src="https://github.com/user-attachments/assets/9d3aa1e4-bb61-428e-a238-af888933b29c" />


## How It Works
The application follows a systematic methodology to ensure accurate and consistent results:
- File Upload: The user uploads a .txt file via the web interface.
- Backend Processing: The Flask backend reads the file content and passes it to each of the five compression functions.
- Performance Measurement: To ensure accuracy and mitigate the effects of system load fluctuations, the compression and decompression times for each algorithm are measured 10 times. The average of these runs is then calculated and displayed.
- Results Display: The calculated metrics (compression ratio, average compression time, average decompression time) are rendered in a dynamic HTML table.
- File Handling: Compressed files are stored temporarily on the server, and download/preview links are generated dynamically for the user.

## Technology Stack
- Backend: Python 3.x, Flask Microframework
- Frontend: HTML5, CSS3
- Core Python Libraries:
- ```zlib``` for the DEFLATE algorithm.
- ```heapq``` for the priority queue in Huffman Coding.
- ```struct``` for packing and unpacking binary data.
- ```time``` for high-resolution performance counters.

## Getting Started Locally
To run this project on your local machine, please follow these steps.

### Clone the Repository
```sh
git clone [https://github.com/your-username/text-compression-analysis.git](https://github.com/your-username/text-compression-analysis.git)
cd text-compression-analysis
```

### Create and Activate a Virtual Environment (Recommended)

# For Windows
```
python -m venv venv
.\venv\Scripts\activate
```

# For macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```
pip install Flask
```
### Run the Application
```
python app.py
```
### Access the Tool
Open your web browser and go to ```http://12.0.0.1:5000```.

## Performance Analysis & Results
The algorithms were benchmarked using a sample text file containing an essay of 9,107 characters. The results below were averaged over 10 consecutive runs.

### Summary of Results
| Method | Compression Ratio | Avg. Compression Time (s) | Avg. Decompression Time (s) | Notes |
| ------ | ------ | ------ | ------ | ------ |
| DEFLATE | 0.4274 | 0.000483 | 0.000127 | Best overall performance in speed and ratio. |
| Arithmetic | 0.6700 | 0.116611 | 0.099052 | Excellent compression but very slow. |
| Huffman | 0.7234 | 0.003939 | 0.011458 | Good speed with moderate compression. |
| LZW | 0.7605 | 0.003421 | 0.002447 | Very fast decompression, good for quick reads. |
| RLE | 2.9150 | 0.006031 | 0.009581 | Ineffective for text; increased the file size. |


### Note: A lower compression ratio is better.

## Graphical Analysis
### Compression Time
<img width="494" height="314" alt="image" src="https://github.com/user-attachments/assets/b5b342c7-710c-4d1f-aedc-3ffd1d1dfea8" />

DEFLATE is exceptionally fast, while Arithmetic Coding's computational complexity makes it significantly slower.

### Decompression Time
<img width="520" height="342" alt="image" src="https://github.com/user-attachments/assets/73f92d8b-0c09-43ea-b25d-d91625e33948" />

DEFLATE and LZW are the fastest for decompression, making them ideal for applications where data is read frequently.

### Compression Ratio
<img width="556" height="359" alt="image" src="https://github.com/user-attachments/assets/dcbfc0d5-b9a7-461d-9c05-9fc72dd28615" />

DEFLATE provides the best space savings. RLE's poor performance on non-repetitive text is clearly visible as it expands the file.

### Overall performance
<img width="567" height="390" alt="image" src="https://github.com/user-attachments/assets/a6b804f7-b315-402f-b007-131966eb1fa3" />

## Conclusion
This analysis confirms that there is no "one-size-fits-all" solution for text compression. The ideal algorithm depends heavily on the specific application requirements:

### Best Overall Performance: DEFLATE is the undisputed winner, offering an excellent balance of high compression ratio and top-tier speed. It is the ideal choice for general-purpose applications.

### Maximum Storage Savings: Arithmetic Coding is the best option when the primary goal is to achieve the smallest possible file size, and processing time is not a major concern (e.g., for archival purposes).

### Lightweight and Fast: Huffman Coding and LZW are excellent choices for systems with limited computational resources or where very low latency is critical.

## Author
### Veditha R (23BCE1301)
