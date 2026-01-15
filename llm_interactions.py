#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based Ecological Interaction Discovery

"""

import json
import re
import os
from typing import List, Tuple, Dict, Optional


try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class LLMInteractionFinder:
    def __init__(self, provider="openai", api_key=None, model=None, debug=True):
        """
        Initialize LLM interaction finder

        Args:
            provider: "openai", "google", or "anthropic"
            api_key: API key for the chosen provider
            model: Specific model to use (optional, defaults to provider's default)
            debug: Enable debug logging
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.debug = debug

        # Set default models if not specified
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-4o"
            elif self.provider == "google":
                self.model = "gemini-1.5-pro"
            elif self.provider == "anthropic":
                self.model = "claude-3-5-sonnet-20241022"

        if self.provider == "openai" and not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        elif self.provider == "google" and not GOOGLE_AVAILABLE:
            raise ImportError("Google AI library not installed. Run: pip install google-generativeai")
        elif self.provider == "anthropic" and not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")


        if self.provider == "openai":
            if api_key:
                self.api_key = api_key
                openai.api_key = api_key
            else:
                env_key = os.getenv("OPENAI_API_KEY")
                if env_key:
                    self.api_key = env_key
                    openai.api_key = env_key
                else:
                    raise ValueError("OpenAI API key required")

        elif self.provider == "google":
            if api_key:
                genai.configure(api_key=api_key)
            else:
                env_key = os.getenv("GOOGLE_API_KEY")
                if env_key:
                    genai.configure(api_key=env_key)
                else:
                    raise ValueError("Google API key required")

        elif self.provider == "anthropic":
            if api_key:
                self.api_key = api_key
            else:
                env_key = os.getenv("ANTHROPIC_API_KEY")
                if env_key:
                    self.api_key = env_key
                else:
                    raise ValueError("Anthropic API key required")
    
    def create_enhanced_prompt(self, variables: List[str], target_variable: str = None, time_resolution: str = None, data_description: str = None) -> str:
        """Create the enhanced prompt for comprehensive ecological interactions with multiple references"""

        variables_str = ", ".join(variables)


        context_info = ""
        if data_description:
            context_info += f"\nDATASET DESCRIPTION: {data_description}"
        if target_variable:
            context_info += f"\nTARGET VARIABLE for prediction: {target_variable}"
        if time_resolution:
            context_info += f"\nTIME RESOLUTION of the data: {time_resolution} (where D=days, W=weeks, M=months, Y=years, H=hours)"


        focus_instruction = ""
        if target_variable:
            focus_instruction = f"\nWhile '{target_variable}' is the target variable for prediction, identify ALL ecologically meaningful interactions in the system - including connections between non-target variables that are part of the broader ecological network."

        prompt = f"""You are an expert ecological and environmental scientist with deep knowledge of peer-reviewed scientific literature.

I am analyzing ecological time series data with the following variables:
{variables_str}{context_info}{focus_instruction}

Please identify known ecological and environmental interactions between these variables based on established peer-reviewed scientific literature. 

CRITICAL: Consider ALL possible interactions in the ecological system:
- Direct effects on the target variable (if specified)
- Interactions between non-target variables (e.g., nutrient cycling, food web relationships)  
- Cascade effects and indirect pathways
- Bidirectional relationships and feedback loops
- Environmental controls affecting multiple variables

For each interaction, provide:
1. Source variable (what influences)
2. Target variable (what is influenced)
3. Interaction strength score (0.1 to 1.0, where 1.0 = very strong, well-established relationship)
4. Time lag (0-10, representing time periods for the effect to manifest{', considering the ' + time_resolution + ' resolution' if time_resolution else ''})
5. Brief scientific justification explaining the ecological mechanism
6. MULTIPLE ACADEMIC REFERENCES (2-4 references when possible):
   - Author names and publication year
   - Complete journal name and article title
   - DOI if available
   - Separate multiple references with " | "

Focus on identifying:
- ALL causal relationships in the ecological network (not just target-focused ones)
- Nutrient cycling interactions (N, P, C cycles)
- Temperature effects on biological and chemical processes
- Food web and trophic interactions
- Environmental controls (light, temperature, hydrology)
- Biogeochemical and physiological mechanisms
- Both direct and indirect ecological pathways

Format your response as a JSON array where each interaction is:
{{
    "source": "variable_name",
    "target": "variable_name", 
    "strength": 0.8,
    "lag": 1,
    "justification": "Brief explanation of ecological mechanism and processes involved",
    "reference": "Reference 1: Author(s), Year. Title. Journal, Volume(Issue), pages. DOI | Reference 2: Author(s), Year. Title. Journal, Volume(Issue), pages. DOI"
}}

REQUIREMENTS:
- Include interactions BETWEEN ALL VARIABLES, not just those involving the target
- Provide 2-4 high-quality academic references per interaction when available
- Separate multiple references with " | " 
- Only include interactions with strong literature support
- If you cannot find multiple quality references, provide at least 1 high-quality reference
- Ensure all citations are accurate and real
- Consider the full ecological network

Variables to analyze: {variables_str}

Return only the JSON array, no other text."""

        return prompt
    
    def query_llm(self, prompt: str) -> str:
        """Query the specified LLM provider"""
        if self.debug:
            print(f"DEBUG: Using {self.provider.upper()} with model {self.model}")
            print(f"DEBUG: Prompt length: {len(prompt)} characters")
            print(f"DEBUG: Prompt preview: {prompt[:300]}...")

        try:
            if self.provider == "openai":
                return self._query_openai(prompt)
            elif self.provider == "google":
                return self._query_google(prompt)
            elif self.provider == "anthropic":
                return self._query_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            if self.debug:
                print(f"DEBUG: LLM query failed: {str(e)}")
            raise
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI GPT model"""
        try:
            # Models that use max_completion_tokens instead of max_tokens
            new_param_models = ['gpt-5', 'gpt-5.2', 'o1-preview', 'o1-mini']
            use_new_param = any(self.model.startswith(m) for m in new_param_models)

            try:
                client = openai.OpenAI(api_key=self.api_key)

                # Build kwargs based on model
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert ecological scientist with extensive knowledge of peer-reviewed scientific literature. Always follow the user's instructions precisely."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }

                # Use appropriate token parameter
                if use_new_param:
                    kwargs["max_completion_tokens"] = 4000
                else:
                    kwargs["max_tokens"] = 4000

                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content.strip()
            except AttributeError:

                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert ecological scientist with extensive knowledge of peer-reviewed scientific literature. Always follow the user's instructions precisely."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _query_google(self, prompt: str) -> str:
        """Query Google Gemini model"""
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.1
                )
            )
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")

    def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude model"""
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                system="You are an expert ecological scientist with extensive knowledge of peer-reviewed scientific literature. Always follow the user's instructions precisely.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract interactions"""
        if self.debug:
            print(f"DEBUG: Response length: {len(response)} characters")
            print(f"DEBUG: Response preview: {response[:500]}...")
        
        try:

            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                if self.debug:
                    print(f"DEBUG: Found JSON of length: {len(json_str)}")
                
                interactions = json.loads(json_str)
                
                if self.debug:
                    print(f"Parsed {len(interactions)} interactions")
                

                cleaned_interactions = []
                for i, interaction in enumerate(interactions):
                    if self.validate_interaction(interaction):
                        normalized = self.normalize_interaction(interaction)
                        cleaned_interactions.append(normalized)
                    elif self.debug:
                        print(f"Interaction {i} failed validation: {interaction}")
                
                if self.debug:
                    print(f"{len(cleaned_interactions)} interactions passed validation")
                
                return cleaned_interactions
            else:
                raise ValueError("No JSON array found in response")
                
        except json.JSONDecodeError as e:
            if self.debug:
                print(f"JSON decode error: {str(e)}")
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            if self.debug:
                print(f"Parse error: {str(e)}")
            raise ValueError(f"Error parsing response: {str(e)}")
    
    def normalize_interaction(self, interaction: Dict) -> Dict:
        """Normalize interaction fields to standard format"""
        normalized = interaction.copy()
        

        if 'strength' in interaction and 'score' not in interaction:
            normalized['score'] = interaction['strength']
        elif 'score' not in interaction and 'strength' not in interaction:
            normalized['score'] = 0.5
        

        if 'justification' not in normalized:
            normalized['justification'] = "Literature-based ecological interaction"
        
        if 'reference' not in normalized:
            normalized['reference'] = ""
        
        return normalized
    
    def validate_interaction(self, interaction: Dict) -> bool:
        """Validate that an interaction has required fields and sensible values"""

        if 'source' not in interaction or 'target' not in interaction:
            return False
        
        score = interaction.get('score', interaction.get('strength', 0))
        if not (0 <= score <= 1):
            return False
        
        lag = interaction.get('lag', 0)
        if not (0 <= lag <= 20):
            return False
        

        if interaction["source"] == interaction["target"]:
            return False
        
        return True
    
    def find_interactions(self, variables: List[str], target_variable: str = None, time_resolution: str = None, data_description: str = None) -> Tuple[List[List], str]:
        """
        Main function to find ecological interactions using LLM with enhanced context

        Args:
            variables: List of variable names
            target_variable: Target variable for prediction (optional)
            time_resolution: Time resolution of data (optional)
            data_description: User's description of the dataset (optional)

        Returns:
            Tuple of (interactions_list, status_message)
            interactions_list format: [[source, target, score, lag, justification_with_reference], ...]
        """
        if self.debug:
            print(f" Finding interactions for variables: {variables}")
            print(f" Target variable: {target_variable}")
            print(f" Time resolution: {time_resolution}")
            print(f" Data description: {data_description}")


        prompt = self.create_enhanced_prompt(variables, target_variable, time_resolution, data_description)
        
        try:

            response = self.query_llm(prompt)
            

            interactions_dict = self.parse_response(response)
            
            interactions_list = []
            for interaction in interactions_dict:

                justification = interaction.get("justification", "Literature-based interaction")
                reference = interaction.get("reference", "")
                
                if reference and reference.strip():
                    full_justification = f"{justification}\n\nReference: {reference}"
                else:
                    if target_variable and time_resolution:
                        full_justification = f"{justification} (target: {target_variable}, resolution: {time_resolution})"
                    else:
                        full_justification = justification
                
                interactions_list.append([
                    interaction["source"],
                    interaction["target"], 
                    float(interaction.get("score", interaction.get("strength", 0.5))),
                    int(interaction.get("lag", 0)),
                    full_justification
                ])
            

            if interactions_list:
                context_info = f" for target: {target_variable}" if target_variable else ""
                status_msg = f"Found {len(interactions_list)} literature-based interactions using {self.provider.upper()}{context_info}"
            else:
                status_msg = f"No confident interactions found using {self.provider.upper()}"
            
            if self.debug:
                print(f" Final status: {status_msg}")
                for i, interaction in enumerate(interactions_list):
                    print(f" Interaction {i}: {interaction[0]} → {interaction[1]} (score: {interaction[2]})")
            
            return interactions_list, status_msg
            
        except Exception as e:
            error_msg = f"Error querying {self.provider.upper()}: {str(e)}"
            if self.debug:
                print(f" Final error: {error_msg}")
            return [], error_msg


def get_llm_interactions(variables: List[str], provider: str = "openai", api_key: str = None,
                        model: str = None, target_variable: str = None, time_resolution: str = None,
                        data_description: str = None, debug: bool = False) -> Tuple[List[List], str]:
    """
    Enhanced convenience function to get LLM interactions with context

    Args:
        variables: List of variable names
        provider: "openai", "google", or "anthropic"
        api_key: API key (optional, will try environment variables)
        model: Specific model to use (optional, defaults to provider's default)
        target_variable: Target variable for prediction (optional)
        time_resolution: Time resolution of data (optional)
        data_description: User's description of the dataset (optional)
        debug: Enable debug logging (optional)

    Returns:
        Tuple of (interactions_list, status_message)
        interactions_list format: [[source, target, score, lag, justification_with_reference], ...]
    """
    try:
        finder = LLMInteractionFinder(provider=provider, api_key=api_key, model=model, debug=debug)
        return finder.find_interactions(variables, target_variable, time_resolution, data_description)
    except Exception as e:
        if debug:
            print(f" Error initializing LLM: {str(e)}")
        return [], f"Error initializing LLM: {str(e)}"


def test_llm_availability():
    """Test function to check LLM availability"""
    results = {
        "openai_available": OPENAI_AVAILABLE,
        "google_available": GOOGLE_AVAILABLE,
        "anthropic_available": ANTHROPIC_AVAILABLE,
        "can_import": True
    }
    return results


if __name__ == "__main__":

    test_results = test_llm_availability()
    print("Enhanced LLM Module Status:")
    print(f"OpenAI available: {test_results['openai_available']}")
    print(f"Google available: {test_results['google_available']}")
    print(f"Anthropic available: {test_results['anthropic_available']}")
    print(f"Module imports correctly: {test_results['can_import']}")

    if test_results['openai_available'] or test_results['google_available'] or test_results['anthropic_available']:
        print("\nModule is ready to use with enhanced features!")
        print("- Full academic references")
        print("- Context-aware prompting")
        print("- Target variable focus")
        print("- Time resolution awareness")
        print("- Data description support")
        print("- System-wide interaction discovery")
        print("- Debug logging available")
    else:
        print("\nInstall required packages:")
        print("pip install openai google-generativeai anthropic")
    

    test_variables = ["Temperature", "Chlorophyll_A", "Nitrate", "Phosphate"]
    test_target = "Chlorophyll_A"
    test_resolution = "2W"
    

    if os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        print(f"\nTesting enhanced functionality with debug enabled...")
        try:
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("GOOGLE_API_KEY"):
                provider = "google"
            else:
                provider = "anthropic"

            interactions, status = get_llm_interactions(
                test_variables,
                provider=provider,
                target_variable=test_target,
                time_resolution=test_resolution,
                data_description="Test marine ecosystem data",
                debug=True
            )
            print(f"Test result: {status}")
            if interactions:
                print(f"Sample interaction: {interactions[0]}")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("\nTo test functionality, set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY environment variable")
