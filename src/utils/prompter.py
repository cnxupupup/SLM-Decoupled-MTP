

def normalize_token(token):
    # token = add_letters_rvq(token)
    if '<sosp>' not in token:
        token = '<sosp>' + token + '<eosp>'
    return token

def normalize_text(text):
    text = text.strip().lower()
    formatted_text = text.capitalize()
    return formatted_text
    
class Prompter(object):

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def generate_prompt(
        self,
        task: str=None,
        audio: str=None,
        text: str=None,
        instruction: str=None,
        response: str=None,
        spk_aware: bool=False,

    ) -> str:
        if task == "asr":
            prompt = "[Human]: Recognize this speech: {audio}.\n [Assistant]: The text is: {text}.\n\n"
            res = prompt.format(audio=audio, text=text)
        elif task == "tts":
            if spk_aware:
                prompt = "[Human]: Read this sentence: {text}. You should speak like: <|placeholder|>.\n [Assistant]: The speech is: {audio}.\n\n"
            else:
                prompt = "[Human]: Read this sentence: {text}.\n [Assistant]: The speech is: {audio}.\n\n"
            
            res = prompt.format(audio=audio, text=text)


        elif task == "text":
            prompt = "[Human]: {instruction}\n [Assistant]: {response}\n\n"
            res = prompt.format(instruction=instruction, response=response)
        return res


    def get_response(self, output: str) -> str:
        return output.split(self.response_split)[1].strip()


