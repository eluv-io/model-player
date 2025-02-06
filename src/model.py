
from typing import List
import numpy as np
from dataclasses import dataclass
from PIL import Image
from unsloth import FastVisionModel
import json

from common_ml.model import FrameModel
from common_ml.tags import FrameTag
from common_ml.types import Data

from src.utils import processPlayerPrediction

from config import config

@dataclass
class RuntimeConfig(Data):
    fps: int
    allow_single_frame: bool
    teams: List[str]

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)
    
class PlayerDetectionModel(FrameModel):
    def __init__(self, model_input_path: str, runtime_config: dict | RuntimeConfig):
        if isinstance(runtime_config, dict):
            runtime_config = RuntimeConfig.from_dict(runtime_config)
        self.model_input_path = model_input_path
        self.config = runtime_config
        self.model, self.tokenizer = self._load_model(model_input_path)
        self.headline = ""
        with open(config['container']['player_info'], 'r') as f:
            self.player_info = json.load(f)
            self.player_info = self._process_player_info(self.player_info, self.config.teams)
        
    def set_headline(self, headline: str):
        self.headline = headline
        
    def tag(self, img: np.ndarray) -> List[FrameTag]:
        img = Image.fromarray(img)
        prompt = self._create_prompt(self.headline)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        print(prompt)
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)

        inputs = self.tokenizer(
            img,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        
        greedy_output = self.model.generate(**inputs, max_new_tokens = 128, use_cache = True)

        res = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        
        res = res.split('assistant')[1].strip()
    
        return [FrameTag.from_dict({"text": res, "confidence": 1.0, "box": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}})]
        
    def _load_model(self, model_path: str):
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = model_path, 
            load_in_4bit = True,
        ) 

        FastVisionModel.for_inference(model)

        return model, tokenizer
    
    def _create_prompt(self, headline: str) -> str:
        prompt = f"Identify all the players in this image. Do not list players who are not in the \"Team - players\" list provided. \n"
        data = {"Headline": headline, "Team - players": self.player_info}
        prompt += json.dumps(data)
        prompt += '\nAlso label the confidence of the prediction as \'HIGHLY\' or \'LESS\' likely.'
        return prompt
    
    def _process_player_info(self, player_data: dict, teams: List[str]) -> dict:
        player_list = dict()
        for player in player_data:
            if player['team'] not in player_list: 
                player_list[player['team']] = [player['name']+'('+player['jersey_number']+')']
            else: 
                player_list[player['team']].append(player['name']+'('+player['jersey_number']+')')
        return {team: players for team, players in player_list.items() if team in teams}
    


class PlayerCaptionModel(FrameModel):
    def __init__(self, model_input_path: str, runtime_config: dict | RuntimeConfig):
        if isinstance(runtime_config, dict):
            runtime_config = RuntimeConfig.from_dict(runtime_config)
        self.model_input_path = model_input_path
        self.config = runtime_config
        self.model, self.tokenizer = self._load_model(model_input_path)
        self.headline = ""
        with open(config['container']['player_map'], 'r') as f:
            self.player_map = json.load(f)
        
        with open(config['container']['player_info'], 'r') as f:
            self.player_info = json.load(f)
            self.player_info = self._process_player_info(self.player_info, self.config.teams)
        
    def set_headline(self, headline: str):
        self.headline = headline
        
    def tag(self, img: np.ndarray) -> List[FrameTag]:
        img = Image.fromarray(img)
        prompt = self._create_prompt(self.headline)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        print(prompt)
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)

        inputs = self.tokenizer(
            img,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        
        greedy_output = self.model.generate(**inputs, max_new_tokens = 128, use_cache = True)

        res = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        
        res = res.split('assistant')[1].strip()
    
        return [FrameTag.from_dict({"text": res, "confidence": 1.0, "box": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}})]
        
    def _load_model(self, model_path: str):
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = model_path, 
            load_in_4bit = True,
        ) 

        FastVisionModel.for_inference(model)

        return model, tokenizer
    
    def _create_prompt(self, headline: str, predicted_players: dict) -> str:
        prompt = f"Provide a caption for the image. The image meta-data is provided below. \n"
        data = {"Headline": headline, "HIGH confidence players": predicted_players['HIGH confidence players'], "LOW confidence players": predicted_players['LOW confidence players']}
        prompt += json.dumps(data)
        prompt += '\nDon\'t output anything except a single line caption.'
        return prompt
    
    def _process_player_details(self, predstr: str):
        processPlayerPrediction(predstr,self.player_map)


    def _process_player_info(self, player_data: dict, teams: List[str]) -> dict:
        player_list = dict()
        for player in player_data:
            if player['team'] not in player_list: 
                player_list[player['team']] = [player['name']+'('+player['jersey_number']+')']
            else: 
                player_list[player['team']].append(player['name']+'('+player['jersey_number']+')')
        return {team: players for team, players in player_list.items() if team in teams}