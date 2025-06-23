# agents/enhanced_base_llm_agent.py
from __future__ import annotations
import logging
import os
import json
import time
import re
from typing import Any, Optional, Type, TypeVar, Dict, List, Union, get_origin, get_args
from abc import ABC
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent
import instructor
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, ValidationError

# Import ModelConfig and model_manager from the config module
from config.modern_models import ModelConfig, model_manager

# Importar sistema de validación JSON y anti-alucinación
from agents.json_validator import TradingSignalValidator, ConstitutionalFinancialPrompt


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

class EnhancedBaseLLMAgent(Agent, ABC):
    """
    VERSIÓN MEJORADA del agente base con:
    - Manejo robusto de errores y timeouts específicos del modelo.
    - Uso directo de raw query si 'supports_tools' es False.
    - Fallbacks automáticos.
    - Validación de tipos mejorada.
    - Logging detallado de respuestas.
    - Mejor handling de respuestas JSON, incluyendo stripping de <think> tags.
    - Creación de defaults más segura para Pydantic models.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore", populate_by_name=True)

    agent_type: str = Field(default="general")
    _llm_model_name: Optional[str] = PrivateAttr(default=None)
    
    _instructor_client: Optional[Any] = PrivateAttr(default=None)
    _raw_client: Optional[OpenAI] = PrivateAttr(default=None)
    _model_config_internal: Optional[ModelConfig] = PrivateAttr(default=None)
    
    _supports_tools: bool = PrivateAttr(default=False)
    _supports_vision: bool = PrivateAttr(default=False)
    _model_timeout: int = PrivateAttr(default=60)

    _performance_stats: Dict[str, Any] = PrivateAttr(default_factory=lambda: {
        'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0,
        'avg_response_time': 0.0, 'last_error': None, 'health_score': 1.0,
        'timeouts': 0, 'connection_errors': 0, 'api_status_errors': 0,
        'validation_errors_parsing':0, 'validation_errors_defaulting':0,
        'instructor_failures': 0, 'raw_query_failures': 0,
        'last_success_time': None, 'last_error_time': None
    })
    _response_log_dir: Path = PrivateAttr(default=Path("logs/llm_responses"))
    _enable_response_logging: bool = PrivateAttr(default=True)
    
    def __init__(self, **kwargs):
        if 'agent_type' not in kwargs:
            class_name = self.__class__.__name__.lower()
            agent_type_map = {
                'sentiment': 'sentiment', 'technical': 'technical',
                'visual': 'visual', 'qabba': 'qabba', 'decision': 'decision'
            }
            for key, value in agent_type_map.items():
                if key in class_name:
                    kwargs['agent_type'] = value
                    break
            else:
                kwargs['agent_type'] = 'general'
        
        # CORRECTED: Defensively remove 'llm_model_name' (public field name) from kwargs
        # before passing to super().__init__ if it was present.
        # This prevents Pydantic from trying to set a field that this class
        # now manages privately as '_llm_model_name'.
        # This is important if the parent Agent class might have 'llm_model_name' as a field.
        kwargs.pop('llm_model_name', None)

        super().__init__(**kwargs)
        
        # Initialize private attributes properly for Pydantic v2
        # Use object.__setattr__ to bypass Pydantic's validation
        object.__setattr__(self, '_performance_stats', {
            'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0,
            'avg_response_time': 0.0, 'last_error': None, 'health_score': 1.0,
            'timeouts': 0, 'connection_errors': 0, 'api_status_errors': 0,
            'validation_errors_parsing': 0, 'validation_errors_defaulting': 0,
            'instructor_failures': 0, 'raw_query_failures': 0,
            'last_success_time': None, 'last_error_time': None
        })
        
        object.__setattr__(self, '_last_context_snapshot', {})
        object.__setattr__(self, '_last_response_raw', "")
        object.__setattr__(self, '_last_reasoning_chain', [])
        object.__setattr__(self, '_debug_info', {})
        
        self._response_log_dir = Path(f"logs/llm_responses/{self.agent_type}")
        self._response_log_dir.mkdir(parents=True, exist_ok=True)
        
        self._safe_setup()

    def _safe_setup(self):
        try:
            self._setup_model_config()
            self._setup_clients()
        except Exception as e:
            logger.error(f"[{self.agent_type}] CRITICAL error in agent setup: {e}", exc_info=True)
            self._emergency_fallback_config()
            self._setup_clients()

    def _setup_model_config(self):
        try:
            if not isinstance(self.agent_type, str) or not self.agent_type:
                logger.error(f"Invalid agent_type: {self.agent_type}. Defaulting to 'general'.")
                self.agent_type = "general"

            self._model_config_internal = model_manager.get_model_config(self.agent_type)
            
            if self._model_config_internal and self._model_config_internal.name != "ollama_model_not_found":
                self._llm_model_name = self._model_config_internal.name
                self._supports_tools = self._model_config_internal.supports_tools
                self._supports_vision = self._model_config_internal.supports_vision
                self._model_timeout = self._model_config_internal.timeout
                logger.info(
                    f"[{self.agent_type}] Configured with model: '{self._llm_model_name}' "
                    f"(Tools: {self._supports_tools}, Vision: {self._supports_vision}, Timeout: {self._model_timeout}s)"
                )
            else:
                logger.warning(f"[{self.agent_type}] No valid model config from manager. Applying emergency fallback.")
                self._emergency_fallback_config()
                       
        except Exception as e:
            logger.error(f"[{self.agent_type}] Error in _setup_model_config: {e}", exc_info=True)
            self._emergency_fallback_config()

    def _emergency_fallback_config(self):
        logger.warning(f"[{self.agent_type}] Activating emergency fallback configuration.")
        self._llm_model_name = "llama3.2:1b"
        self._supports_tools = False
        self._supports_vision = False
        self._model_timeout = 30
        self._model_config_internal = ModelConfig(
            name=self._llm_model_name,
            supports_tools=self._supports_tools,
            supports_vision=self._supports_vision,
            context_length=4096,
            best_for=["emergency_fallback"],
            temperature=0.5,
            max_tokens=512,
            timeout=self._model_timeout
        )

    def _setup_clients(self):
        if not self._llm_model_name or self._llm_model_name == "ollama_model_not_found":
            logger.error(f"[{self.agent_type}] Cannot setup clients: LLM model name is not set or invalid ('{self._llm_model_name}').")
            self._raw_client = None
            self._instructor_client = None
            return
            
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")

            logger.info(f"[{self.agent_type}] Setting up OpenAI client for model '{self._llm_model_name}' with timeout: {self._model_timeout}s")
            self._raw_client = OpenAI(
                base_url=f"{ollama_url}/v1",
                api_key=api_key,
                timeout=self._model_timeout
            )
            
            # Only setup Instructor if tools are supported and not known problematic models
            if self._supports_tools and self._raw_client and not self._is_known_problematic_model():
                try:
                    self._instructor_client = instructor.from_openai(self._raw_client)
                    logger.info(f"[{self.agent_type}] Instructor client setup for '{self._llm_model_name}'.")
                except Exception as e_instr:
                    logger.warning(f"[{self.agent_type}] Instructor setup failed for '{self._llm_model_name}': {e_instr}. Will rely on raw queries.")
                    self._instructor_client = None
                    self._supports_tools = False  # Disable tools if Instructor setup fails
            else:
                self._instructor_client = None
                if self._supports_tools:
                     logger.warning(f"[{self.agent_type}] Tools supported by config, but raw client failed or model known to be problematic. Instructor client not set up.")

        except Exception as e:
            logger.error(f"[{self.agent_type}] Client setup failed for '{self._llm_model_name}': {e}", exc_info=True)
            self._raw_client = None
            self._instructor_client = None

    def _is_known_problematic_model(self) -> bool:
        """Check if the model is known to have issues with Instructor/tools."""
        if not self._llm_model_name:
            return True
        
        # Known problematic models that don't work well with Instructor
        problematic_patterns = [
            "nous-hermes2pro",
            "adrienbrault/nous-hermes2pro",
            "registry.ollama.ai/adrienbrault/nous-hermes2pro",
            "qwen2.5",  # qwen2.5 models have multiple tool calls issues
            "qwen2.5:7b-instruct",
            "qwen2.5:7b-instruct-q5_k_m"
        ]
        
        model_name_lower = self._llm_model_name.lower()
        return any(pattern in model_name_lower for pattern in problematic_patterns)
    
    def _log_llm_interaction(self, prompt_summary: str, response_summary: str, duration: float, error: Optional[str] = None, success: bool = True):
        if not self._enable_response_logging:
            return
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self._response_log_dir / f"{timestamp_str}.json"
        
        log_data = {
            "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
            "agent_type": self.agent_type,
            "model_used": self._llm_model_name,
            "prompt_summary": prompt_summary[:1000],
            "response_summary": response_summary[:5000] if response_summary else None,
            "duration_seconds": round(duration, 3),
            "error_message": error[:1000] if error else None,
            "was_successful": success,
            "performance_snapshot": {
                "health_score": self._performance_stats['health_score'],
                "total_requests": self._performance_stats['total_requests'],
                "timeouts": self._performance_stats['timeouts']
            }
        }
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[{self.agent_type}] Failed to write LLM interaction log: {e}")

    def _query_llm(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None,
        retry_count: int = 2
    ) -> Optional[T]:
        if not self._raw_client:
            logger.error(f"[{self.agent_type}] No LLM client available for query. Model: '{self._llm_model_name}'.")
            self._record_failure("No LLM client")
            self._log_llm_interaction(prompt, "No client", 0.0, "No LLM client", False)
            return self._create_safe_default(response_model, "LLM client not initialized.")

        start_time = time.time()
        self._performance_stats['total_requests'] += 1
        
        temp_to_use = temperature if temperature is not None else getattr(self._model_config_internal, 'temperature', 0.2)
        tokens_to_use = max_tokens if max_tokens is not None else getattr(self._model_config_internal, 'max_tokens', 1024)
        
        last_exception = None
        response_obj: Optional[T] = None

        for attempt in range(retry_count + 1):
            logger.info(f"[{self.agent_type}] LLM query attempt {attempt + 1}/{retry_count + 1} for model '{self._llm_model_name}'.")
            try:
                if self._instructor_client and self._supports_tools:
                    logger.debug(f"[{self.agent_type}] Attempting Instructor query.")
                    response_obj = self._try_instructor_query(prompt, response_model, temp_to_use, tokens_to_use, images)
                    if response_obj:
                        logger.info(f"[{self.agent_type}] Instructor query successful on attempt {attempt + 1}.")
                        break
                    else:
                        logger.warning(f"[{self.agent_type}] Instructor query returned None on attempt {attempt + 1}. Will try raw query or retry.")
                        self._performance_stats['instructor_failures'] +=1
                        
                logger.debug(f"[{self.agent_type}] Attempting raw query.")
                response_obj = self._try_raw_query_improved(prompt, response_model, temp_to_use, tokens_to_use, images)
                if response_obj:
                    logger.info(f"[{self.agent_type}] Raw query successful on attempt {attempt + 1}.")
                    break
                else:
                    logger.warning(f"[{self.agent_type}] Raw query returned None on attempt {attempt + 1}. Retrying if attempts remain.")
                    self._performance_stats['raw_query_failures'] +=1
                    
            except APITimeoutError as e:
                last_exception = e
                self._performance_stats['timeouts'] += 1
                logger.warning(f"[{self.agent_type}] LLM Query Attempt {attempt + 1} timed out: {e}")
            except APIConnectionError as e:
                last_exception = e
                self._performance_stats['connection_errors'] += 1
                logger.error(f"[{self.agent_type}] LLM Query Attempt {attempt + 1} connection error: {e}")
            except APIStatusError as e:
                last_exception = e
                self._performance_stats['api_status_errors'] += 1
                logger.error(f"[{self.agent_type}] LLM Query Attempt {attempt + 1} API status error: {e.status_code} - {e.message}")
                if e.status_code == 400 and "does not support tools" in e.message.lower():
                    logger.warning(f"[{self.agent_type}] Model '{self._llm_model_name}' reported as not supporting tools by Ollama. Disabling _supports_tools for this agent instance.")
                    self._supports_tools = False
                    self._instructor_client = None
            except ValidationError as e:
                last_exception = e
                self._performance_stats['validation_errors_parsing'] += 1
                logger.error(f"[{self.agent_type}] Pydantic validation error parsing LLM response on attempt {attempt + 1}: {e}")
            except Exception as e:
                last_exception = e
                logger.error(f"[{self.agent_type}] Unexpected error during LLM query attempt {attempt + 1}: {e}", exc_info=True)
            
            if response_obj:
                break

            if attempt < retry_count:
                sleep_duration = 1.5 * (2 ** attempt)
                logger.info(f"[{self.agent_type}] Retrying in {sleep_duration:.1f} seconds...")
                time.sleep(sleep_duration)
        
        duration = time.time() - start_time
        prompt_summary = f"{prompt[:150]}..." if len(prompt) > 150 else prompt
        
        if response_obj:
            self._record_success(duration)
            self._log_llm_interaction(prompt_summary, str(response_obj.model_dump_json(indent=2))[:1000], duration, success=True)
            return response_obj
        else:
            error_msg = f"All {retry_count + 1} LLM query attempts failed. Last error: {type(last_exception).__name__} - {str(last_exception)[:200]}"
            logger.error(f"[{self.agent_type}] {error_msg}")
            self._record_failure(error_msg)
            self._log_llm_interaction(prompt_summary, "API Error", duration, error_msg, success=False)
            return self._create_safe_default(response_model, error_msg)

    def _try_instructor_query(self, prompt: str, response_model: Type[T],
                            temp: float, tokens: int, images: Optional[List[str]]) -> Optional[T]:
        if not self._instructor_client:
            logger.debug(f"[{self.agent_type}] Instructor client not available, skipping instructor query.")
            return None
        try:
            messages: List[Dict[str, Any]] = [{"role": "user", "content": []}]
            content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

            if images and self._supports_vision:
                for img_b64 in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
            elif images and not self._supports_vision:
                 logger.warning(f"[{self.agent_type}] Images provided but model '{self._llm_model_name}' does not support vision. Ignoring images.")

            messages[0]["content"] = content_parts
            
            # Type guard to ensure model name is not None
            if not self._llm_model_name:
                logger.error(f"[{self.agent_type}] Model name is None when trying Instructor query")
                return None
            
            response = self._instructor_client.chat.completions.create(
                model=self._llm_model_name,
                messages=messages,
                response_model=response_model,
                temperature=temp,
                max_tokens=tokens,
                max_retries=0  # Disable retries to avoid multiple tool call issues
            )
            return response # type: ignore
            
        except APIStatusError as e:
            if "does not support tools" in str(e.message).lower():
                logger.warning(f"[{self.agent_type}] Instructor query failed: Model '{self._llm_model_name}' reported by API as not supporting tools. Disabling for this agent instance.")
                self._supports_tools = False
                self._instructor_client = None
            else:
                logger.warning(f"[{self.agent_type}] Instructor query APIStatusError: {e.status_code} - {e.message}")
            self._performance_stats['instructor_failures'] +=1
            raise
        except Exception as e:
            error_msg = str(e).lower()
            # Handle multiple tool calls error specifically
            if "multiple tool calls" in error_msg or "instructor does not support multiple tool calls" in error_msg:
                logger.info(f"[{self.agent_type}] Model '{self._llm_model_name}' doesn't support Instructor tool calls. Switching to raw query mode (this is expected behavior).")
                self._supports_tools = False
                self._instructor_client = None
                return None  # Return None to trigger fallback to raw query
            
            logger.warning(f"[{self.agent_type}] Instructor query failed with unexpected error: {e}", exc_info=True)
            self._performance_stats['instructor_failures'] +=1
            if isinstance(e, (APITimeoutError, APIConnectionError)):
                raise
            return None

    def _try_raw_query_improved(self, prompt: str, response_model: Type[T],
                               temp: float, tokens: int, images: Optional[List[str]]) -> Optional[T]:
        if not self._raw_client:
             logger.error(f"[{self.agent_type}] Raw client not available for raw query.")
             return None
        try:
            schema_json = json.dumps(response_model.model_json_schema(), indent=2)
            example_json = json.dumps(self._generate_example_from_schema(response_model.model_json_schema()), indent=2)
            
            json_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS FOR RESPONSE FORMATTING:
1. You MUST respond with ONLY a valid JSON object that strictly adheres to the provided schema.
2. Do NOT include any other text, explanations, apologies, or markdown formatting (like ```json ... ```) outside of the JSON object itself.
3. The entire response must be the JSON object.

JSON Schema to follow:
{schema_json}

Example of the exact JSON output format required:
{example_json}

Now, provide ONLY the JSON object based on the analysis:"""

            messages: List[Dict[str, Any]] = [{"role": "user", "content": []}]
            content_parts: List[Dict[str, Any]] = [{"type": "text", "text": json_prompt}]

            if images and self._supports_vision:
                for img_b64 in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
            elif images and not self._supports_vision:
                 logger.warning(f"[{self.agent_type}] Images provided to raw query but model '{self._llm_model_name}' does not support vision. Ignoring images.")

            messages[0]["content"] = content_parts
            
            response_raw = self._raw_client.chat.completions.create(
                model=self._llm_model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens
            )
            
            content_text = response_raw.choices[0].message.content
            if not content_text:
                logger.warning(f"[{self.agent_type}] Raw query returned empty content.")
                return None
            
            parsed_json = self._robust_json_parse_improved(content_text)
            if parsed_json:
                try:
                    return response_model(**parsed_json)
                except ValidationError as e:
                    self._performance_stats['validation_errors_parsing'] += 1
                    logger.error(f"[{self.agent_type}] Pydantic validation error for raw query response: {e}. Parsed JSON: {parsed_json}")
                    logger.debug(f"[{self.agent_type}] Problematic LLM content (raw): '{content_text[:500]}...'")
                    return None
            else:
                logger.warning(f"[{self.agent_type}] Robust JSON parsing failed for raw query. Content: '{content_text[:200]}...'")
                return None
            
        except Exception as e:
            logger.error(f"[{self.agent_type}] Error during _try_raw_query_improved: {e}", exc_info=True)
            if isinstance(e, (APITimeoutError, APIConnectionError, APIStatusError)):
                raise
            return None


    def _robust_json_parse_improved(self, content: str) -> Optional[Dict[Any, Any]]:
        if not content:
            return None
            
        original_content = content
        
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.MULTILINE)
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            json_candidate = content[first_brace : last_brace + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"[{self.agent_type}] Could not parse JSON from LLM response. Original content (first 200 chars): '{original_content[:200]}...'")
        return None

    def _generate_example_from_schema(self, schema: Dict[Any, Any]) -> Dict[Any, Any]:
        example: Dict[Any, Any] = {}
        properties = schema.get('properties', {})
        
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get('type')
            prop_enum = prop_schema.get('enum')
            prop_default = prop_schema.get('default')

            if prop_default is not None:
                example[prop_name] = prop_default
            elif prop_enum:
                example[prop_name] = prop_enum[0]
            elif prop_type == 'string':
                example[prop_name] = f"example_{prop_name}"
            elif prop_type == 'integer':
                example[prop_name] = 1
            elif prop_type == 'number':
                example[prop_name] = 1.0
            elif prop_type == 'boolean':
                example[prop_name] = True
            elif prop_type == 'array':
                items_schema = prop_schema.get('items', {})
                item_type = items_schema.get('type')
                if item_type == 'string': example[prop_name] = [f"example_item_1"]
                elif item_type == 'integer': example[prop_name] = [1]
                elif item_type == 'number': example[prop_name] = [1.0]
                else: example[prop_name] = []
            elif prop_type == 'object':
                example[prop_name] = {}
            else:
                example[prop_name] = None
        return example

    def _create_safe_default(self, response_model: Type[T], reason: str) -> Optional[T]:
        logger.warning(f"[{self.agent_type}] Creating default for {response_model.__name__} due to: {reason[:150]}")
        defaults: Dict[str, Any] = {}
        
        try:
            for field_name, field_info in response_model.model_fields.items():
                if field_info.default is not None:
                    defaults[field_name] = field_info.default
                    continue
                if field_info.default_factory is not None:
                    defaults[field_name] = field_info.default_factory()
                    continue

                is_optional = get_origin(field_info.annotation) is Union and type(None) in get_args(field_info.annotation)
                if is_optional:
                    defaults[field_name] = None
                    continue

                field_type = field_info.annotation
                if field_type == str: defaults[field_name] = f"Default value due to error: {reason[:50]}"
                elif field_type == int: defaults[field_name] = 0
                elif field_type == float: defaults[field_name] = 0.0
                elif field_type == bool: defaults[field_name] = False
                elif field_type == list or get_origin(field_type) is list: defaults[field_name] = []
                elif field_type == dict or get_origin(field_type) is dict: defaults[field_name] = {}
                elif field_name in ['reasoning', 'combined_reasoning', 'reasoning_short']:
                    defaults[field_name] = reason[:250]
                elif ('signal' in field_name.lower() or 'assessment' in field_name.lower() or 'action' in field_name.lower()) and 'Literal' in str(field_info.annotation):
                    literal_args = get_args(field_info.annotation)
                    non_none_literal = next((arg for arg in literal_args if arg is not None), None)
                    if non_none_literal is not None:
                        defaults[field_name] = non_none_literal
                    elif literal_args:
                        defaults[field_name] = literal_args[0]
                    else:
                        # Special handling for known literal fields
                        if 'assessment' in field_name.lower():
                            defaults[field_name] = "UNCLEAR"
                        elif 'action' in field_name.lower():
                            defaults[field_name] = "AVOID_TRADE"
                        else:
                            defaults[field_name] = "DEFAULT_SIGNAL_ERROR"
                        logger.error(f"Could not determine default for Literal field {field_name}, using fallback")
                elif 'confidence' in field_name.lower() and (isinstance(field_type, type) and (field_type is float or field_type is type(None))):
                    defaults[field_name] = 0.0
                else:
                    logger.warning(f"[{self.agent_type}] No default strategy for required field '{field_name}' of type {field_type} in {response_model.__name__}. Validation might fail.")
            
            return response_model(**defaults)
        except ValidationError as e:
            self._performance_stats['validation_errors_defaulting'] +=1
            logger.error(f"[{self.agent_type}] CRITICAL: Pydantic validation error while CREATING SAFE DEFAULT for {response_model.__name__}: {e}. Defaults tried: {defaults}")
            return None
        except Exception as e:
            logger.error(f"[{self.agent_type}] Unexpected error creating safe default for {response_model.__name__}: {e}", exc_info=True)
            return None

    def _record_success(self, response_time: float):
        self._performance_stats['successful_requests'] += 1
        total_successful = self._performance_stats['successful_requests']
        current_avg_time = self._performance_stats['avg_response_time']
        self._performance_stats['avg_response_time'] = \
            ((current_avg_time * (total_successful - 1)) + response_time) / total_successful if total_successful > 0 else response_time
        self._performance_stats['last_success_time'] = time.time()
        self._update_health_score()

    def _record_failure(self, error_summary: str):
        self._performance_stats['failed_requests'] += 1
        self._performance_stats['last_error'] = error_summary[:200]
        self._performance_stats['last_error_time'] = time.time()
        self._update_health_score()

    def _update_health_score(self):
        total_req = self._performance_stats['total_requests']
        if total_req == 0:
            self._performance_stats['health_score'] = 1.0
            return

        success_rate = self._performance_stats['successful_requests'] / total_req
        
        error_time_factor = 1.0
        if self._performance_stats['last_error_time']:
            time_since_last_error = time.time() - self._performance_stats['last_error_time']
            if time_since_last_error < 300:
                error_time_factor = max(0.1, 1.0 - ( (300 - time_since_last_error) / 300) * 0.5 )
        
        timeout_penalty = 1.0 - min(0.5, (self._performance_stats['timeouts'] / total_req) * 0.2)

        self._performance_stats['health_score'] = round(success_rate * error_time_factor * timeout_penalty, 3)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        total_req = self._performance_stats['total_requests']
        success_rate_calc = (self._performance_stats['successful_requests'] / total_req * 100) if total_req > 0 else 0.0
        return {
            'agent_type': self.agent_type,
            'model_used': self._llm_model_name,
            'health_score': self._performance_stats['health_score'],
            'total_requests': total_req,
            'successful_requests': self._performance_stats['successful_requests'],
            'failed_requests': self._performance_stats['failed_requests'],
            'success_rate_percentage': round(success_rate_calc, 2),
            'avg_response_time_seconds': round(self._performance_stats['avg_response_time'], 3),
            'timeouts': self._performance_stats['timeouts'],
            'connection_errors': self._performance_stats['connection_errors'],
            'api_status_errors': self._performance_stats['api_status_errors'],
            'validation_errors_parsing': self._performance_stats['validation_errors_parsing'],
            'validation_errors_defaulting': self._performance_stats['validation_errors_defaulting'],
            'instructor_failures': self._performance_stats['instructor_failures'],
            'raw_query_failures': self._performance_stats['raw_query_failures'],
            'last_error_summary': self._performance_stats['last_error'],
            'last_error_timestamp': datetime.fromtimestamp(self._performance_stats['last_error_time']).isoformat() if self._performance_stats['last_error_time'] else None,
            'last_success_timestamp': datetime.fromtimestamp(self._performance_stats['last_success_time']).isoformat() if self._performance_stats['last_success_time'] else None,
        }

    # =========================
    # JSON VALIDATION & ANTI-HALLUCINATION INTEGRATION
    # =========================
    
    def _query_llm_with_validation(
        self,
        prompt: str,
        response_model: Type[T],
        schema_type: str = "trading",
        use_constitutional: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None,
        retry_count: int = 2
    ) -> Optional[T]:
        """
        Query LLM with integrated JSON validation and anti-hallucination techniques.
        
        Args:
            prompt: Base prompt
            response_model: Pydantic model for response
            schema_type: Schema type for JSON validation ("trading", "qabba", "visual")
            use_constitutional: Whether to use constitutional prompting
            temperature: Model temperature
            max_tokens: Max tokens
            images: Base64 images for vision models
            retry_count: Number of retries
            
        Returns:
            Validated response model or None if all validation fails
        """
        # Apply constitutional prompting if enabled
        if use_constitutional:
            constitutional_prompt = ConstitutionalFinancialPrompt.create_constitutional_prompt(
                prompt, self.agent_type
            )
        else:
            constitutional_prompt = prompt
            
        # Use the existing _query_llm method
        response = self._query_llm(
            constitutional_prompt, 
            response_model, 
            temperature, 
            max_tokens, 
            images, 
            retry_count
        )
        
        if response is None:
            logger.warning(f"[{self.agent_type}] LLM query returned None, attempting JSON validation fallback")
            return self._create_safe_default(response_model, "LLM query failed completely")
            
        # Validate the response using our JSON validator
        try:
            validator = TradingSignalValidator()
            response_dict = response.model_dump()
            
            # Validate with our robust validator
            validated_data = validator.validate_and_repair(
                json.dumps(response_dict), 
                schema_type
            )
            
            if validated_data is None:
                logger.warning(f"[{self.agent_type}] JSON validation failed, using original response")
                return response
                
            # Create new validated response
            try:
                validated_response = response_model(**validated_data)
                logger.info(f"[{self.agent_type}] Response successfully validated and potentially repaired")
                return validated_response
            except ValidationError as e:
                logger.warning(f"[{self.agent_type}] Failed to create validated response model: {e}")
                return response
                
        except Exception as e:
            logger.error(f"[{self.agent_type}] Error during JSON validation: {e}")
            return response
    
    def _get_schema_type_for_agent(self) -> str:
        """Determine appropriate schema type based on agent type."""
        schema_mapping = {
            "sentiment": "sentiment",
            "technical": "trading", 
            "visual": "visual",
            "qabba": "qabba",
            "decision": "trading"
        }
        return schema_mapping.get(self.agent_type, "trading")
        
    def _apply_anti_hallucination_checks(self, response: T) -> T:
        """
        Apply anti-hallucination checks specific to financial analysis.
        
        Args:
            response: The response model to check
            
        Returns:
            Modified response with anti-hallucination safeguards
        """
        try:
            response_dict = response.model_dump()
            
            # Check for overly confident predictions without sufficient data
            if hasattr(response, 'confidence') or hasattr(response, 'confidence_score'):
                confidence_field = getattr(response, 'confidence', None) or getattr(response, 'confidence_score', None)
                if confidence_field and confidence_field > 0.9:
                    # High confidence should be justified
                    reasoning = getattr(response, 'reasoning', "")
                    if len(reasoning) < 50:  # Insufficient reasoning for high confidence
                        logger.warning(f"[{self.agent_type}] High confidence ({confidence_field}) with insufficient reasoning, reducing confidence")
                        if hasattr(response, 'confidence'):
                            response.confidence = min(0.8, confidence_field)
                        if hasattr(response, 'confidence_score'):
                            response.confidence_score = min(0.8, confidence_field)
            
            # Add disclaimer for predictions
            if hasattr(response, 'reasoning'):
                reasoning = response.reasoning
                if not any(disclaimer in reasoning.lower() for disclaimer in 
                          ["potential", "uncertain", "risk", "no guarantee", "may", "could", "might"]):
                    # Add uncertainty disclaimer
                    response.reasoning = f"{reasoning}. Note: This analysis carries inherent uncertainty and should not be considered guaranteed financial advice."
            
            return response
            
        except Exception as e:
            logger.error(f"[{self.agent_type}] Error applying anti-hallucination checks: {e}")
            return response

