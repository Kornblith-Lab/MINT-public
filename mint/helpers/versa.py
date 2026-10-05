from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Type, Union
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from pydantic import BaseModel
from openai import AzureOpenAI
import os

from sqlitedict import SqliteDict
from tqdm.auto import tqdm

load_dotenv(find_dotenv())
API_KEY: str = os.environ.get("API_KEY", "")
API_VERSION = "2025-04-01-preview"
RESOURCE_ENDPOINT = os.environ.get("RESOURCE_ENDPOINT", "")

assert len(API_KEY) == 88, f"API_KEY of length {len(API_KEY)} is not correct"
assert RESOURCE_ENDPOINT != "" 
assert API_VERSION != ""

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

@dataclass
class ExtractionConfig:
    prompt_file: Path|str
    response_schema: Union[Type[BaseModel], dict]
    csv_file: Optional[Path|str] = None
    df: Optional[pd.DataFrame] = None
    note_col: str = "note_text"
    llm: str = "gpt-5-nano-2025-08-07"
    debug: Optional[int] = None
    tool_calling: bool = False
    n_replicates: int = 1
    n_workers: int = 100

    def __post_init__(self):
        if self.tool_calling:
            assert type(self.response_schema) == dict

        assert self.csv_file is not None or self.df is not None, "You must provide either a CSV or a dataframe as input"
        assert self.n_replicates > 0, "n_replicates must be greater than 0"

class Extraction:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.prompt_text = Path(self.config.prompt_file).read_text()
        # print(self.prompt_text)
    def extract(self, output_path: Path|str):
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        df = self.config.df.copy().reset_index() if self.config.df is not None else pd.read_csv(self.config.csv_file) # type: ignore
        full_df = df.copy()
        db = SqliteDict(output_path.with_suffix(".sqlite"), tablename="inference", autocommit=True)

        input_token_count = 0
        output_token_count = 0

        if self.config.debug:
            df = df.head(self.config.debug)
            full_df = df.copy()

        already_done = df[df["EncounterKey"].apply(lambda k: f"{k}_rep0" in db)]
        if len(already_done) > 0:
            remaining = len(df) - len(already_done)
            resp = input(f"{len(already_done)} encounters already done. Skip them and process only {remaining} remaining? (yes/y): ")
            if resp.strip().lower() in ("yes", "y"):
                df = df[~df.index.isin(already_done.index)]
            else:
                print("Aborting.")
                return None

        if len(df) > 0:
            # Submit futures for all replicates
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as ex:
                futures = []
                for idx, row in tqdm(df.iterrows()):
                    for replicate_num in range(self.config.n_replicates):
                        futures.append(ex.submit(self._extract_single, idx, row, replicate_num))

                pbar = tqdm(as_completed(futures), total=len(futures))

                for future in pbar:
                    try:
                        completion, idx, row, replicate_num = future.result()
                        if not self.config.tool_calling:
                            response = completion.choices[0].message

                            if response.refusal: raise Exception(f"Model refused: {response.refusal}")
                            if not response.parsed: raise Exception("No parsed output")

                            json_text = response.parsed.model_dump_json()
                        else:
                            response = completion.choices[0].message.tool_calls # type: ignore
                            if response is None: raise Exception(f"Model error")
                            json_text = response[0].function.arguments

                        db[f"{str(row.EncounterKey)}_rep{replicate_num}"] = json_text

                        input_token_count += completion.usage.prompt_tokens # type: ignore
                        output_token_count += completion.usage.completion_tokens # type: ignore
                        pbar.set_description(f"Total input {input_token_count:,} tokens and {output_token_count:,} output tokens used")
                    except Exception as e:
                        print(f"Error: {e}")

        # Rebuild inference column for the full df from the db
        full_df["inference"] = full_df["EncounterKey"].apply(
            lambda k: db.get(f"{k}_rep0", None)
        )
        full_df["inference"] = full_df["inference"].apply(
            lambda text: json.loads(text) if isinstance(text, str) else {}
        )
        inference_df = pd.json_normalize(full_df["inference"]) # type: ignore
        full_df = pd.concat([full_df.drop(columns=["inference"]), inference_df], axis=1)
        full_df.to_csv(output_path.with_suffix(".csv"), index=False)
        db.close()
        return full_df
    
    def _extract_single(self, idx, row: pd.Series, replicate_num: int):
        messages = [
            { "role": "system", "content": self.prompt_text },
            { "role": "user", "content": row[self.config.note_col] }
        ]

        if not self.config.tool_calling:
            completion = client.chat.completions.parse(
                model=self.config.llm,
                reasoning_effort="medium",
                messages=messages,
                response_format=self.config.response_schema, # type: ignore
            )
        else:
            completion = client.chat.completions.create(
                model=self.config.llm,
                messages=messages,
                reasoning_effort="medium",
                n=1,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "clinical_extraction",
                            "description": "redacted",
                            "strict": True,
                            "parameters": self.config.response_schema
                        }
                    } # type: ignore
                ], # type: ignore
                tool_choice="required"
            )

        if self.config.debug:
            print("PROMPT " * 10)
            print(self.prompt_text)
            print("=" * 20)
            print("NOTE " * 10)
            print(row[self.config.note_col])
            print("=" * 20)
            print("RESPONSE " * 10)
            print(completion.choices[0].message)
            print("=" * 20)

        return completion, idx, row, replicate_num
