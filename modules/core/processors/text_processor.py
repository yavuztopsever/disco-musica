"""Text Processor for handling music generation prompts."""

import re
import json
from typing import Dict, List, Optional, Union, Set, Any
from pathlib import Path
import logging
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import defaultdict

from ..exceptions.base_exceptions import (
    ProcessingError,
    ValidationError
)


class TextProcessor:
    """Processor for handling text prompts for music generation.
    
    This class provides functionality for processing text prompts,
    including tokenization, normalization, musical term extraction,
    and semantic analysis.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize the text processor.
        
        Args:
            model_name: Name of the pretrained model to use.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise ProcessingError(f"Failed to initialize text processor: {e}")
            
        # Load musical terms
        self._load_musical_terms()
        
        # Text cleaning patterns
        self.cleaning_patterns = [
            (r"\s+", " "),  # Normalize whitespace
            (r"[^\w\s\-.,!?]", ""),  # Remove special characters
            (r"\s*([.,!?])\s*", r"\1 "),  # Normalize punctuation spacing
            (r"\s+", " ")  # Final whitespace cleanup
        ]
        
    def _load_musical_terms(self) -> None:
        """Load musical term dictionaries."""
        # These would typically be loaded from files
        self.musical_terms = {
            "genres": {
                "classical", "jazz", "rock", "electronic", "ambient",
                "orchestral", "chamber", "symphony", "concerto"
            },
            "moods": {
                "happy", "sad", "energetic", "calm", "dramatic",
                "peaceful", "intense", "melancholic", "uplifting"
            },
            "instruments": {
                "piano", "violin", "guitar", "drums", "bass",
                "trumpet", "flute", "cello", "synthesizer"
            },
            "dynamics": {
                "loud", "soft", "crescendo", "diminuendo",
                "forte", "piano", "fortissimo", "pianissimo"
            },
            "tempos": {
                "fast", "slow", "moderate", "allegro", "adagio",
                "andante", "presto", "largo", "accelerando"
            }
        }
        
    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_tensors: bool = True
    ) -> Union[List[int], torch.Tensor]:
        """Tokenize text input.
        
        Args:
            text: Input text.
            max_length: Optional maximum sequence length.
            return_tensors: Whether to return PyTorch tensors.
            
        Returns:
            Tokenized text as list or tensor.
            
        Raises:
            ProcessingError: If tokenization fails.
        """
        try:
            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=max_length,
                padding="max_length" if max_length else False,
                truncation=bool(max_length),
                return_tensors="pt" if return_tensors else None
            )
            
            return tokens["input_ids"] if return_tensors else tokens["input_ids"][0]
            
        except Exception as e:
            raise ProcessingError(f"Error tokenizing text: {e}")
            
    def get_embeddings(
        self,
        text: str,
        pooling: str = "mean"
    ) -> np.ndarray:
        """Get text embeddings.
        
        Args:
            text: Input text.
            pooling: Pooling method ("mean" or "cls").
            
        Returns:
            Text embedding array.
            
        Raises:
            ValidationError: If pooling method is invalid.
            ProcessingError: If embedding fails.
        """
        if pooling not in ["mean", "cls"]:
            raise ValidationError(f"Invalid pooling method: {pooling}")
            
        try:
            # Tokenize
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                embeddings = outputs.last_hidden_state
                
                if pooling == "mean":
                    # Mean pooling
                    attention_mask = tokens["attention_mask"]
                    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    masked_embeddings = embeddings * mask
                    summed = torch.sum(masked_embeddings, dim=1)
                    counts = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
                    pooled = summed / counts
                else:
                    # CLS token pooling
                    pooled = embeddings[:, 0]
                
            return pooled.numpy()
            
        except Exception as e:
            raise ProcessingError(f"Error getting embeddings: {e}")
            
    def normalize_text(self, text: str) -> str:
        """Normalize text input.
        
        Args:
            text: Input text.
            
        Returns:
            Normalized text.
        """
        normalized = text.lower().strip()
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            normalized = re.sub(pattern, replacement, normalized)
            
        return normalized.strip()
        
    def extract_musical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract musical terms from text.
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary of extracted terms by category.
        """
        # Normalize text
        normalized = self.normalize_text(text)
        words = set(normalized.split())
        
        # Extract terms by category
        extracted = defaultdict(list)
        for category, terms in self.musical_terms.items():
            matches = words.intersection(terms)
            if matches:
                extracted[category] = sorted(matches)
                
        return dict(extracted)
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment for mood matching.
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary of sentiment scores.
            
        Raises:
            ProcessingError: If analysis fails.
        """
        try:
            # Get embeddings
            embeddings = self.get_embeddings(text)
            
            # Simple sentiment analysis using embedding dimensions
            sentiment = {
                "valence": float(np.mean(embeddings[:, :256])),  # Positive/negative
                "arousal": float(np.mean(embeddings[:, 256:512])),  # Energy level
                "dominance": float(np.mean(embeddings[:, 512:]))  # Intensity
            }
            
            # Normalize to [0, 1] range
            for key in sentiment:
                sentiment[key] = (sentiment[key] + 1) / 2
                
            return sentiment
            
        except Exception as e:
            raise ProcessingError(f"Error analyzing sentiment: {e}")
            
    def get_key_phrases(
        self,
        text: str,
        max_phrases: int = 5
    ) -> List[str]:
        """Extract key phrases from text.
        
        Args:
            text: Input text.
            max_phrases: Maximum number of phrases to extract.
            
        Returns:
            List of key phrases.
        """
        # Normalize text
        normalized = self.normalize_text(text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', normalized)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract noun phrases (simple approach)
        phrases = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 2:
                phrases.append(" ".join(words[:3]))
            elif words:
                phrases.append(sentence)
                
        return phrases[:max_phrases]
        
    def augment_prompt(
        self,
        text: str,
        style_terms: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """Augment prompt with musical terms.
        
        Args:
            text: Input text.
            style_terms: Optional additional style terms by category.
            
        Returns:
            Augmented prompt.
        """
        # Extract existing terms
        existing_terms = self.extract_musical_terms(text)
        
        # Combine with provided terms
        if style_terms:
            for category, terms in style_terms.items():
                if category in existing_terms:
                    existing_terms[category].extend(terms)
                else:
                    existing_terms[category] = terms
                    
        # Build augmented prompt
        prompt_parts = [text.strip()]
        
        for category, terms in existing_terms.items():
            if terms:
                # Add category-specific phrases
                if category == "genres":
                    prompt_parts.append(f"in the style of {', '.join(terms)}")
                elif category == "moods":
                    prompt_parts.append(f"with a {', '.join(terms)} mood")
                elif category == "instruments":
                    prompt_parts.append(f"featuring {', '.join(terms)}")
                elif category == "dynamics":
                    prompt_parts.append(f"with {', '.join(terms)} dynamics")
                elif category == "tempos":
                    prompt_parts.append(f"at a {', '.join(terms)} tempo")
                    
        return " ".join(prompt_parts)
        
    def get_memory_usage(self) -> float:
        """Get processor memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        # Estimate memory usage of internal data structures
        memory = 0
        
        # Musical terms dictionaries
        for terms in self.musical_terms.values():
            memory += sum(len(term) for term in terms)
            
        # Model parameters (if loaded)
        if hasattr(self, "model"):
            memory += sum(
                param.nelement() * param.element_size()
                for param in self.model.parameters()
            )
            
        return memory 