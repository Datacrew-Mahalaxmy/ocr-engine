"""
preprocess.py - Image preprocessing for better OCR
Applies various OpenCV techniques to enhance text extraction
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Apply preprocessing to images before OCR
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Main preprocessing pipeline
        """
        # Convert PIL to OpenCV
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing steps
        img = self._denoise(img)
        img = self._enhance_contrast(img)
        img = self._sharpen(img)
        img = self._binarize(img)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def _denoise(self, img):
        """Remove noise while preserving edges"""
        # Fast NL Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return denoised
    
    def _enhance_contrast(self, img):
        """Improve contrast for better text visibility"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def _sharpen(self, img):
        """Sharpen image to make text crisper"""
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return sharpened
    
    def _binarize(self, img):
        """Convert to binary (black and white)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3-channel
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary_bgr


class AdaptivePreprocessor:
    """
    Adaptive preprocessing based on image quality
    """
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        
    def analyze_quality(self, image: Image.Image) -> dict:
        """Analyze image quality to determine preprocessing needs"""
        img = np.array(image.convert('L'))
        
        # Calculate contrast
        contrast = img.std()
        
        # Calculate brightness
        brightness = img.mean()
        
        # Detect noise level
        edges = cv2.Canny(img, 50, 150)
        noise_level = np.sum(edges) / (img.shape[0] * img.shape[1])
        
        return {
            'contrast': contrast,
            'brightness': brightness,
            'noise_level': noise_level,
            'needs_enhancement': contrast < 40,
            'needs_denoising': noise_level > 0.1
        }
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Apply adaptive preprocessing based on quality analysis"""
        quality = self.analyze_quality(image)
        
        logger.debug(f"Image quality: contrast={quality['contrast']:.1f}, "
                    f"noise={quality['noise_level']:.3f}")
        
        # Convert PIL to OpenCV
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing based on quality
        if quality['needs_denoising']:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            logger.debug("Applied denoising")
        
        if quality['needs_enhancement']:
            # Enhance contrast
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            logger.debug("Applied contrast enhancement")
        
        # Always sharpen slightly
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * 0.5
        img = cv2.filter2D(img, -1, kernel)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


# Specialized preprocessors for different document types
class AadhaarPreprocessor:
    """Specialized for Aadhaar cards (colored background, small text)"""
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (helps with colored backgrounds)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convert back to 3-channel
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)


class BankStatementPreprocessor:
    """Specialized for bank statements (tabular data)"""
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Morphological operations to enhance table lines
        kernel = np.ones((1, 3), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            eroded, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2
        )
        
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)


def get_preprocessor(doc_type: str = 'default'):
    """Factory function to get appropriate preprocessor"""
    preprocessors = {
        'aadhaar_card': AadhaarPreprocessor(),
        'bank_statement': BankStatementPreprocessor(),
        'default': AdaptivePreprocessor()
    }
    return preprocessors.get(doc_type, preprocessors['default'])