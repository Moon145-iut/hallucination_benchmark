from typing import Dict, Any
import re
import spacy
import numpy as np
from nltk.corpus import stopwords
from llm.free_llm import generate_text

class HallucinationScorer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.stop_words = set(stopwords.words('english'))
        
    def calculate_hallucination_score(self, 
                                    knowledge: str, 
                                    true_response: str, 
                                    generated_response: str) -> Dict[str, Any]:
        # Stage 1: LLM-based Classification
        llm_score = self._get_llm_classification(knowledge, true_response, generated_response)
        
        # Stage 2: Algorithmic Adjustments
        entity_mismatch = self._check_entity_mismatch(true_response, generated_response)
        fact_contradiction = self._check_fact_contradiction(knowledge, generated_response)
        semantic_drift = self._calculate_semantic_drift(true_response, generated_response)
        
        # Calculate weighted algorithmic score (0-2 scale)
        algo_score = (
            entity_mismatch * 0.4 +  # Entity errors weight
            fact_contradiction * 0.4 +  # Factual contradiction weight
            semantic_drift * 0.2  # Semantic drift weight
        ) * 2  # Scale to 0-2

        if llm_score < 0:
            llm_score = algo_score
        
        # Combine LLM and algorithmic scores with confidence weighting
        confidence = 1 - np.std([llm_score/2, algo_score/2])  # Normalize to 0-1
        final_score = (llm_score * 0.6 + algo_score * 0.4)  # Weight LLM score higher
        final_score = max(0.0, min(2.0, final_score))
        
        # Categorize final severity
        severity = self._categorize_severity(final_score)
        
        return {
            "score": round(final_score, 2),
            "confidence": round(confidence, 2),
            "severity": severity,
            "components": {
                "llm_score": llm_score,
                "algorithmic_score": round(algo_score, 2),
                "entity_mismatch": round(entity_mismatch, 2),
                "fact_contradiction": round(fact_contradiction, 2),
                "semantic_drift": round(semantic_drift, 2)
            }
        }
    
    def _get_llm_classification(self, knowledge: str, true_response: str, generated_response: str) -> float:
        """Use a free instruction-tuned LLM to classify hallucination severity"""
        prompt = f"""Analyze the following response for hallucination severity on a scale of 0-2:
        
Knowledge Context: {knowledge}
True Response: {true_response}
Generated Response: {generated_response}

Score using:
0 = Factual (no hallucination)
1 = Slightly incorrect (minor/subtle hallucination) 
2 = Strongly hallucinated (major errors/fabrications)

Provide score as a single number (0, 1, or 2)."""

        try:
            response = generate_text(
                prompt,
                temperature=0.0,
                max_new_tokens=32,
            )
            match = re.search(r"\b([0-2])\b", response)
            if match:
                score = float(match.group(1))
                return float(max(0, min(2, score)))
            raise ValueError(f"Unrecognized LLM score: {response}")
        except Exception as exc:
            print(f"LLM scoring failed: {exc}")
            return -1.0  # Signal failure to get LLM score
    
    def _check_entity_mismatch(self, true_response: str, generated_response: str) -> float:
        """Calculate entity mismatch ratio between responses"""
        true_ents = set([ent.text.lower() for ent in self.nlp(true_response).ents])
        gen_ents = set([ent.text.lower() for ent in self.nlp(generated_response).ents])
        
        if not true_ents:
            return 0.0
            
        mismatched = len(true_ents.symmetric_difference(gen_ents))
        total = len(true_ents.union(gen_ents))
        
        return mismatched / total if total > 0 else 0.0
    
    def _check_fact_contradiction(self, knowledge: str, generated_response: str) -> float:
        """Check factual contradictions against knowledge"""
        knowledge_doc = self.nlp(knowledge)
        knowledge_facts = set([sent.text.lower() for sent in knowledge_doc.sents])
        
        contradiction_score = 0.0
        response_doc = self.nlp(generated_response)
        
        for sent in response_doc.sents:
            sent_text = sent.text.lower()
            contradictions = [
                self._calculate_contradiction(sent_text, fact) 
                for fact in knowledge_facts
            ]
            contradiction_score += max(contradictions) if contradictions else 0
            
        return min(contradiction_score / len(list(response_doc.sents)), 1.0)
    
    def _calculate_semantic_drift(self, true_response: str, generated_response: str) -> float:
        """Calculate semantic similarity between responses"""
        true_doc = self.nlp(true_response)
        gen_doc = self.nlp(generated_response)

        similarity = true_doc.similarity(gen_doc)
        drift = 1 - similarity
        return max(0.0, min(1.0, drift))  # Convert to drift score
    
    def _calculate_contradiction(self, sent1: str, sent2: str) -> float:
        """Calculate contradiction score between two sentences"""
        doc1 = self.nlp(sent1)
        doc2 = self.nlp(sent2)
        
        similarity = doc1.similarity(doc2)
        return 1 - similarity if similarity < 0.5 else 0.0
    
    def _categorize_severity(self, score: float) -> str:
        """Categorize the hallucination severity based on score"""
        if score < 0.5:
            return "factual"
        elif score < 1.2:
            return "slightly_incorrect"
        else:
            return "strongly_hallucinated"
