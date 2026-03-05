/* FileUpload.js - React component for file upload with drag-and-drop support */

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFileSelect, disabled }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    },
    maxSize: 100 * 1024 * 1024,
    multiple: false
  });

  return (
    <div 
      {...getRootProps()} 
      className={`dropzone ${isDragActive ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
    >
      <input {...getInputProps()} />
      <div className="dropzone-content">
        <span className="upload-icon">📄</span>
        {isDragActive ? (
          <p>Drop the file here...</p>
        ) : (
          <>
            <p>Drag & drop a file here</p>
            <p className="small">or click to browse</p>
            <p className="hint">PDF, PNG, JPG, JPEG, TIFF, BMP (max 100MB)</p>
          </>
        )}
      </div>
    </div>
  );
};

export default FileUpload;