import os
import time
import zlib
from flask import Flask, request, render_template, send_from_directory, make_response, session
import struct

app = Flask(__name__)
app.secret_key = 'your_secret_key' 
UPLOAD_FOLDER = 'uploads'
COMPRESSED_FOLDER = 'compressed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

# RLE Compression
def rle_compress(data):
    if not data:
        return b''
        
    result = bytearray()
    count = 1
    current = data[0]
    
    for char in data[1:]:
        if char == current and count < 255:
            count += 1
        else:
            # Convert count to single byte and character to 2 bytes
            result.append(count)
            result.extend(current.encode('utf-16-le'))
            current = char
            count = 1
    
    # Handle the last run
    result.append(count)
    result.extend(current.encode('utf-16-le'))
    return bytes(result)

def rle_decompress(data):
    if not data:
        return ""
    
    result = []
    i = 0
    while i < len(data):
        count = data[i]
        # Each character takes 2 bytes in UTF-16-LE
        char = data[i+1:i+3].decode('utf-16-le')
        result.append(char * count)
        i += 3  # Move to next group (1 byte count + 2 bytes char)
    
    return "".join(result)

# Huffman Compression (Simplified Version)
import heapq
from collections import Counter, namedtuple

class Node(namedtuple("Node", ["char", "freq", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    frequency = Counter(text)
    heap = [Node(char, freq, None, None) for char, freq in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, Node(None, left.freq + right.freq, left, right))
    return heap[0] if heap else None

def build_codes(node, prefix='', codebook={}):
    if node is not None:
        if node.char is not None:
            codebook[node.char] = prefix
        build_codes(node.left, prefix + '0', codebook)
        build_codes(node.right, prefix + '1', codebook)
    return codebook

def huffman_compress(data):
    tree = build_huffman_tree(data)
    codebook = build_codes(tree, prefix='', codebook={})  # Create new codebook
    
    # Convert codebook to bytes
    codebook_str = str(codebook)
    codebook_bytes = codebook_str.encode('utf-8')
    codebook_size = len(codebook_bytes).to_bytes(4, byteorder='big')
    
    # Compress the actual data
    encoded_data = ''.join(codebook[char] for char in data)
    padding = 8 - len(encoded_data) % 8 if len(encoded_data) % 8 != 0 else 0
    padded_encoded = encoded_data + '0' * padding
    
    # Convert binary string to bytes
    data_bytes = bytearray([padding])
    for i in range(0, len(padded_encoded), 8):
        data_bytes.append(int(padded_encoded[i:i+8], 2))
    
    # Combine all components
    return codebook_size + codebook_bytes + bytes(data_bytes)

def huffman_decompress(byte_data):
    # Extract codebook size and codebook
    codebook_size = int.from_bytes(byte_data[:4], byteorder='big')
    codebook_bytes = byte_data[4:4+codebook_size]
    codebook = eval(codebook_bytes.decode('utf-8'))
    
    # Get compressed data
    data_bytes = byte_data[4+codebook_size:]
    padding = data_bytes[0]
    
    # Convert bytes to binary string
    bits = ''
    for byte in data_bytes[1:]:
        bits += format(byte, '08b')
    
    # Remove padding
    if padding:
        bits = bits[:-padding]
    
    # Create reverse codebook
    reverse_codebook = {code: char for char, code in codebook.items()}
    
    # Decode
    decoded = ''
    current = ''
    for bit in bits:
        current += bit
        if current in reverse_codebook:
            decoded += reverse_codebook[current]
            current = ''
    
    return decoded

# LZW Compression
def lzw_compress(unicode_text):
    if not unicode_text:
        return struct.pack('>H', 0)
    
    # Initialize dictionary with all possible single-byte characters
    dictionary = {chr(i): i for i in range(256)}
    dict_size = 256
    
    w = ''
    result = []
    
    for char in unicode_text:
        wc = w + char
        if wc in dictionary:
            w = wc
        else:
            if w in dictionary:  # Only append if the sequence exists
                result.append(dictionary[w])
            if dict_size < 65536:  # Limit dictionary size to 2-byte codes
                dictionary[wc] = dict_size
                dict_size += 1
            w = char
    
    # Output the last code
    if w and w in dictionary:  # Check if w exists in dictionary
        result.append(dictionary[w])
    
    if not result:  # Handle case where no compression occurred
        return struct.pack('>H', ord(unicode_text[0]))
        
    return struct.pack(f'>{len(result)}H', *result)

def lzw_decompress(compressed_data):
    # Unpack the 2-byte integers
    result = struct.unpack(f'>{len(compressed_data)//2}H', compressed_data)
    
    if not result:
        return ""
    
    # Initialize dictionary with single characters
    dictionary = {i: chr(i) for i in range(256)}
    dict_size = 256
    
    # Initialize with first character
    w = dictionary[result[0]]
    decoded = w
    
    for code in result[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Invalid compressed data")
        
        decoded += entry
        
        # Add new sequence to dictionary
        if dict_size < 65536:  # Limit dictionary size to 2-byte codes
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
        
        w = entry
    
    return decoded

# Arithmetic Coding
def get_frequencies(data):
    freq = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1
    return freq

import ast

def arithmetic_compress(data):
    if not data:
        return b''

    freq = get_frequencies(data)
    freq_list = sorted(freq.items(), key=lambda x: x[0])
    total = sum(freq.values())

    # Store frequency list
    freq_bytes = str(freq_list).encode('utf-8')
    freq_size = len(freq_bytes).to_bytes(4, 'big')
    data_len = len(data).to_bytes(4, 'big')

    low = 0
    high = (1 << 64) - 1
    full_range = 1 << 64
    quarter = full_range // 4
    half = quarter * 2
    three_quarter = quarter * 3

    bits = []
    pending = 0

    for char in data:
        range_ = high - low + 1

        cum_freq = 0
        for c, f in freq_list:
            if c == char:
                break
            cum_freq += f

        char_freq = dict(freq_list)[char]
        high = low + (range_ * (cum_freq + char_freq)) // total - 1
        low = low + (range_ * cum_freq) // total

        while True:
            if high < half:
                bits.append(0)
                bits.extend([1] * pending)
                pending = 0
                low = low * 2
                high = high * 2 + 1
            elif low >= half:
                bits.append(1)
                bits.extend([0] * pending)
                pending = 0
                low = (low - half) * 2
                high = (high - half) * 2 + 1
            elif low >= quarter and high < three_quarter:
                pending += 1
                low = (low - quarter) * 2
                high = (high - quarter) * 2 + 1
            else:
                break

    pending += 1
    if low < quarter:
        bits.append(0)
        bits.extend([1] * pending)
    else:
        bits.append(1)
        bits.extend([0] * pending)

    # Convert bits to bytes
    bitstring = ''.join(map(str, bits))
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += '0' * padding

    out_bytes = bytearray([padding])
    for i in range(0, len(bitstring), 8):
        byte = int(bitstring[i:i+8], 2)
        out_bytes.append(byte)

    return freq_size + freq_bytes + data_len + out_bytes

def arithmetic_decompress(data):
    if not data:
        return ""

    freq_size = int.from_bytes(data[:4], 'big')
    freq_list = ast.literal_eval(data[4:4 + freq_size].decode('utf-8'))
    freq_list = sorted(freq_list, key=lambda x: x[0])
    total = sum(f for _, f in freq_list)
    char_map = dict(freq_list)

    pos = 4 + freq_size
    length = int.from_bytes(data[pos:pos + 4], 'big')
    pos += 4

    padding = data[pos]
    bits = ''
    for byte in data[pos+1:]:
        bits += format(byte, '08b')
    bits = bits[:len(bits) - padding]

    # Setup
    low = 0
    high = (1 << 64) - 1
    full_range = 1 << 64
    quarter = full_range // 4
    half = quarter * 2
    three_quarter = quarter * 3

    value = int(bits[:64], 2)
    bit_index = 64
    result = []

    for _ in range(length):
        range_ = high - low + 1
        scaled = ((value - low + 1) * total - 1) // range_

        cum_freq = 0
        for char, freq in freq_list:
            if cum_freq + freq > scaled:
                result.append(char)
                high = low + (range_ * (cum_freq + freq)) // total - 1
                low = low + (range_ * cum_freq) // total
                break
            cum_freq += freq

        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                value -= half
            elif low >= quarter and high < three_quarter:
                low -= quarter
                high -= quarter
                value -= quarter
            else:
                break

            low *= 2
            high = high * 2 + 1
            if bit_index < len(bits):
                value = value * 2 + int(bits[bit_index])
                bit_index += 1
            else:
                value = value * 2

    return ''.join(result)



def save_compressed_file(method, data):
    filename = f"{method}.bin"
    path = os.path.join(COMPRESSED_FOLDER, filename)
    with open(path, 'wb') as f:
        f.write(data)
    return filename

# Add these decompression functions after the compression functions



def lzw_decompress(compressed_data):
    # Unpack the 2-byte integers
    result = struct.unpack(f'>{len(compressed_data)//2}H', compressed_data)
    
    # Initialize dictionary with single characters
    dictionary = {idx: chr(idx) for idx in range(256)}
    dict_size = 256
    
    # Initialize with first character
    w = chr(result[0])
    decoded = w
    
    # Process remaining codes
    for code in result[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Invalid compressed data")
            
        decoded += entry
        
        # Add new sequence to dictionary
        if dict_size < 65536:  # Limit dictionary size to 2-byte codes
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
            
        w = entry
    
    return decoded

# Modify the compress_and_measure function
def compress_and_measure(method_name, func, decomp_func, text):
    # Compression
    start_comp = time.perf_counter()
    compressed = func(text)
    end_comp = time.perf_counter()
    
    # Decompression
    start_decomp = time.perf_counter()
    if method_name == 'deflate':
        decompressed = decomp_func(compressed).decode('utf-8')
    else:
        decompressed = decomp_func(compressed)
    end_decomp = time.perf_counter()
    
    file_name = save_compressed_file(method_name, compressed)
    ratio = len(compressed) / len(text.encode('utf-8'))
    decomp_time = end_decomp - start_decomp
    
    # Format both ratio and decompression time to avoid scientific notation
    if method_name in ['deflate', 'arithmetic']:
        ratio = format(ratio, '.4f')
        decomp_time = format(decomp_time, '.6f')
    else:
        ratio = round(ratio, 4)
        decomp_time = round(decomp_time, 6)
    
    return {
        'method': method_name,
        'ratio': ratio,
        'comp_time': round(end_comp - start_comp, 6),
        'decomp_time': decomp_time,
        'file': file_name,
        'decompressed': decompressed[:100] + '...' if len(decompressed) > 100 else decompressed
    }

# Modify the route to use the new compression/decompression pairs
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Clear session if new file is uploaded
        session.clear()
        
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            results = []
            results.append(compress_and_measure('rle', rle_compress, rle_decompress, text))
            results.append(compress_and_measure('huffman', huffman_compress, huffman_decompress, text))
            results.append(compress_and_measure('lzw', lzw_compress, lzw_decompress, text))
            results.append(compress_and_measure('deflate', 
                                            lambda d: zlib.compress(d.encode('utf-8')), 
                                            zlib.decompress, 
                                            text))
            results.append(compress_and_measure('arithmetic', 
                                            arithmetic_compress, 
                                            arithmetic_decompress, 
                                            text))
            session['results'] = results

            return render_template('index.html', results=session['results'])
    return render_template('index.html')
@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(COMPRESSED_FOLDER, filename, as_attachment=True)

@app.route('/preview/<method>')
def preview_text(method):
    # Find the most recent compressed file for this method
    filename = f"{method}.bin"
    compressed_path = os.path.join(COMPRESSED_FOLDER, filename)
    
    if not os.path.exists(compressed_path):
        return "File not found", 404
        
    with open(compressed_path, 'rb') as f:
        compressed_data = f.read()
    
    # Decompress based on method
    if method == 'rle':
        decompressed = rle_decompress(compressed_data)
    elif method == 'huffman':
        decompressed = huffman_decompress(compressed_data)
    elif method == 'lzw':
        decompressed = lzw_decompress(compressed_data)
    elif method == 'deflate':
        decompressed = zlib.decompress(compressed_data).decode('utf-8')
    elif method == 'arithmetic':
        decompressed = arithmetic_decompress(compressed_data)
    else:
        return "Invalid method", 400
        
    # Return the decompressed text in a simple HTML page
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decompressed Text - {method.upper()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            pre {{ white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <h2>Decompressed Text ({method.upper()})</h2>
        <pre>{decompressed}</pre>
    </body>
    </html>
    """

@app.route('/download_decompressed/<method>')
def download_decompressed(method):
    filename = f"{method}.bin"
    compressed_path = os.path.join(COMPRESSED_FOLDER, filename)
    
    if not os.path.exists(compressed_path):
        return "File not found", 404
        
    with open(compressed_path, 'rb') as f:
        compressed_data = f.read()
    
    # Decompress based on method
    if method == 'rle':
        decompressed = rle_decompress(compressed_data)
    elif method == 'huffman':
        decompressed = huffman_decompress(compressed_data)
    elif method == 'lzw':
        decompressed = lzw_decompress(compressed_data)
    elif method == 'deflate':
        decompressed = zlib.decompress(compressed_data).decode('utf-8')
    elif method == 'arithmetic':
        decompressed = arithmetic_decompress(compressed_data)
    else:
        return "Invalid method", 400
    
    # Create response with decompressed text
    response = make_response(decompressed)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = f'attachment; filename={method}_decompressed.txt'
    
    return response

@app.route('/clear_session')
def clear_session():
    session.clear()
    return "Session cleared successfully", 200

if __name__ == '__main__':
    app.run(debug=True)