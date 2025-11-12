import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

load_dotenv()

class CropRecommendation(BaseModel):
    crop_name: str = Field(description="Recommended crop name")
    planting_season: str = Field(description="Best planting season")
    care_instructions: List[str] = Field(description="Care and maintenance tips")
    expected_yield: str = Field(description="Expected yield information")
    market_value: str = Field(description="Current market value/demand")

    @field_validator("crop_name", mode="before")
    def clean_crop_name(cls, v):
        if isinstance(v, dict):
            return v.get("description", str(v))
        return str(v)

class GreenCureAI:
    def __init__(self, api_key_name=None):
        api_key_mapping = {
            "GROQ1": "GROQ_API_KEY_1",
            "GROQ2": "GROQ_API_KEY_2",
            "GROQ3": "GROQ_API_KEY_3",
            "GROQ4": "GROQ_API_KEY_4",
        }
        api_key_value = None
        if api_key_name and api_key_name in api_key_mapping:
            env_var_name = api_key_mapping[api_key_name]
            api_key_value = os.getenv(env_var_name)
            if not api_key_value:
                raise ValueError(f"API key not found for {api_key_name}")

        self.llm = ChatGroq(
            api_key=api_key_value,
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )

    def get_crop_recommendation(
        self, location: str, soil_type: str, season: str, farm_size: str
    ) -> CropRecommendation:
        parser = PydanticOutputParser(pydantic_object=CropRecommendation)
        prompt = PromptTemplate(
            template=(
                "As an agricultural expert specializing in Indian farming, provide ONE SINGLE crop recommendation for:\n"
                "Location: {location}\n"
                "Soil Type: {soil_type}\n"
                "Season: {season}\n"
                "Farm Size: {farm_size}\n\n"
                "Consider Indian agricultural conditions, monsoon patterns, and local market demands.\n"
                "Return ONLY ONE crop recommendation in this EXACT JSON format:\n\n"
                "{\n"
                '  "crop_name": "Name of the SINGLE most suitable crop",\n'
                '  "planting_season": "Best time to plant this crop with specific months",\n'
                '  "care_instructions": ["Detailed instruction 1", "Detailed instruction 2", "Detailed instruction 3"],\n'
                '  "expected_yield": "Realistic yield per acre as a simple string",\n'
                '  "market_value": "Current market price and demand as a simple string"\n'
                "}\n\n"
                "IMPORTANT: Return only ONE crop, not a list. Market value should be a simple string, not an object.\n"
                "Focus on the MOST suitable crop for the given conditions.\n"
                "{format_instructions}"
            ),
            input_variables=["location", "soil_type", "season", "farm_size"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.llm.invoke(
                    prompt.format(
                        location=location,
                        soil_type=soil_type,
                        season=season,
                        farm_size=farm_size,
                    )
                )
                content = response.content.strip()
                if content.startswith("["):
                    import json
                    try:
                        json_array = json.loads(content)
                        if isinstance(json_array, list) and len(json_array) > 0:
                            first = json_array[0]
                            if isinstance(first.get("market_value"), dict):
                                info = first["market_value"]
                                price = info.get("current_price", "Price varies")
                                demand = info.get("demand", "Good demand")
                                first["market_value"] = f"{price}, {demand}"
                            content = json.dumps(first)
                    except Exception:
                        pass

                result = parser.parse(content)
                if not result.crop_name or not result.care_instructions:
                    raise ValueError("Invalid crop recommendation format")
                return result

            except Exception:
                if attempt == max_attempts - 1:
                    return CropRecommendation(
                        crop_name="Wheat",
                        planting_season="Rabi season (November-December) for your region",
                        care_instructions=[
                            "Prepare field with proper ploughing and leveling",
                            "Apply organic manure 15-20 tons per hectare",
                            "Maintain proper irrigation schedule",
                        ],
                        expected_yield="25-30 quintals per hectare",
                        market_value="₹2000-2500 per quintal with good market demand",
                    )
                continue
