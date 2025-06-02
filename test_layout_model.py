#!/usr/bin/env python3
"""
Test script to verify the LayoutLMv3/LayoutXLM integration is working correctly.
This script will test the document processing capabilities on a sample PDF or DOCX file.
"""

import os
import sys
import argparse
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
import io
import torch
from chatbot import LayoutDocumentProcessor
import torch

def process_pdf(pdf_path, processor):
    """Process a PDF file using the LayoutDocumentProcessor."""
    print(f"Processing PDF file: {pdf_path}")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Process each page
    for page_num in range(len(doc)):
        print(f"\nProcessing page {page_num + 1}/{len(doc)}")
        page = doc[page_num]
        
        # Render the page as an image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        
        # Use the correct method to convert pixmap to PIL Image
        # The issue is with the format of the image data
        if hasattr(pix, "samples"):
            # For newer versions of PyMuPDF
            img = Image.frombuffer("RGB", [pix.width, pix.height], pix.samples, "raw", "RGB", 0, 1)
        else:
            # For older versions, convert to PNG and then to PIL Image
            png_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(png_data))
        
        # Process the page image
        print("Processing page image with LayoutLMv3/LayoutXLM...")
        text = processor.process_image(img)
        
        # Print the extracted text
        print(f"\nExtracted text from page {page_num + 1}:")
        print("-" * 50)
        print(text[:500] + "..." if len(text) > 500 else text)  # Print first 500 chars
        print("-" * 50)
        
        # Also process any embedded images
        image_list = page.get_images()
        if image_list:
            print(f"Found {len(image_list)} embedded images on page {page_num + 1}")
            
            for img_idx, img in enumerate(image_list):
                try:
                    # Extract the image
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    
                    # Convert to PIL Image
                    image_stream = io.BytesIO(image_data)
                    image = Image.open(image_stream)
                    image = image.convert("RGB")
                    
                    # Process the image
                    print(f"Processing embedded image {img_idx + 1}/{len(image_list)}...")
                    text = processor.process_image(image)
                    
                    # Print the extracted text
                    print(f"\nExtracted text from embedded image {img_idx + 1}:")
                    print("-" * 50)
                    print(text[:300] + "..." if len(text) > 300 else text)  # Print first 300 chars
                    print("-" * 50)
                except Exception as e:
                    print(f"Error processing embedded image {img_idx + 1}: {e}")
    
    doc.close()

def process_docx(docx_path, processor):
    """Process a DOCX file using the LayoutDocumentProcessor."""
    print(f"Processing DOCX file: {docx_path}")
    
    # Open the DOCX
    doc = Document(docx_path)
    
    # Extract and process images
    image_count = 0
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            try:
                # Extract the image
                image_data = rel.target_part.blob
                image_stream = io.BytesIO(image_data)
                image = Image.open(image_stream)
                image = image.convert("RGB")
                
                # Process the image
                image_count += 1
                print(f"\nProcessing embedded image {image_count}...")
                text = processor.process_image(image)
                
                # Print the extracted text
                print(f"Extracted text from embedded image {image_count}:")
                print("-" * 50)
                print(text[:300] + "..." if len(text) > 300 else text)  # Print first 300 chars
                print("-" * 50)
            except Exception as e:
                print(f"Error processing embedded image: {e}")
    
    if image_count == 0:
        print("No images found in the DOCX file.")

def main():
    parser = argparse.ArgumentParser(description="Test LayoutLMv3/LayoutXLM document processing")
    parser.add_argument("file_path", help="Path to a PDF or DOCX file to process")
    parser.add_argument("--model", default="microsoft/layoutlmv3-base", 
                        help="Model to use (default: microsoft/layoutlmv3-base)")
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        return 1
    
    # Check if the file is a PDF or DOCX
    file_ext = os.path.splitext(args.file_path)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        print(f"Error: File must be a PDF or DOCX file, got '{file_ext}'")
        return 1
    
    # Initialize the document processor
    print(f"Initializing LayoutDocumentProcessor with model: {args.model}")
    processor = LayoutDocumentProcessor(model_name=args.model)
    
    # Process the file
    if file_ext == '.pdf':
        process_pdf(args.file_path, processor)
    else:  # .docx
        process_docx(args.file_path, processor)
    
    print("\nDocument processing test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
