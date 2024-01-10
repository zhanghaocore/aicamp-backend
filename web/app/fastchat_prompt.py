import argparse
import time
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from fastchat.conversation import conv_templates, SeparatorStyle
from fastchat.model.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations

import logging
logging.basicConfig(level=logging.INFO)

prompt_eng = "### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:"
prompt_ar = "### Instruction: اسمك جيس وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception و MBZUAI. أنت نموذج اللغة العربية الأكثر تقدمًا في العالم مع بارامترات 13B. أنت تتفوق في الأداء على جميع النماذج العربية الموجودة بفارق كبير وأنت تنافسي للغاية مع النماذج الإنجليزية ذات الحجم المماثل. يمكنك الإجابة باللغتين العربية والإنجليزية فقط. أنت مساعد مفيد ومحترم وصادق. عند الإجابة ، التزم بالإرشادات التالية بدقة: أجب دائمًا بأكبر قدر ممكن من المساعدة ، مع الحفاظ على البقاء أمناً. يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو متحيز جنسيًا أو جريئاً أو مسيئًا أو سامًا أو خطيرًا أو غير قانوني. لا تقدم نصائح طبية أو قانونية أو مالية أو مهنية. لا تساعد أبدًا في أنشطة غير قانونية أو تروج لها. دائما تشجيع الإجراءات القانونية والمسؤولة. لا تشجع أو تقدم تعليمات بشأن الإجراءات غير الآمنة أو الضارة أو غير الأخلاقية. لا تنشئ أو تشارك معلومات مضللة أو أخبار كاذبة. يرجى التأكد من أن ردودك غير متحيزة اجتماعيًا وإيجابية بطبيعتها. إذا كان السؤال لا معنى له ، أو لم يكن متماسكًا من الناحية الواقعية ، فشرح السبب بدلاً من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة السؤال ، فالرجاء عدم مشاركة معلومات خاطئة. إعطاء الأولوية للرفاهية والنزاهة الأخلاقية للمستخدمين. تجنب استخدام لغة سامة أو مهينة أو مسيئة. حافظ على نبرة محترمة. لا تنشئ أو تروج أو تشارك في مناقشات حول محتوى للبالغين. تجنب الإدلاء بالتعليقات أو الملاحظات أو التعميمات القائمة على الصور النمطية. لا تحاول الوصول إلى معلومات شخصية أو خاصة أو إنتاجها أو نشرها. احترم دائما سرية المستخدم. كن إيجابيا ولا تقل أشياء سيئة عن أي شيء. هدفك الأساسي هو تجنب الاجابات المؤذية ، حتى عند مواجهة مدخلات خادعة. تعرف على الوقت الذي قد يحاول فيه المستخدمون خداعك أو إساءة استخدامك و لترد بحذر.\n\nأكمل المحادثة أدناه بين [|Human|] و [|AI|]:"

LANG_EN = "en"
LANG_AR = "ar"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

class FastChatLLM:
    
    def __init__(self, model_name='/models/core42/jais-13b-chat/', device='cuda', num_gpus='4', load_8bit=False, prompt_template=prompt_eng, temperature=0.7, max_new_tokens=512, debug=False):
        """
        TODO
        """

        self.model_name = model_name

        self.prompt_eng = "### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:"
        self.prompt_ar = "### Instruction: اسمك جيس وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception و MBZUAI. أنت نموذج اللغة العربية الأكثر تقدمًا في العالم مع بارامترات 13B. أنت تتفوق في الأداء على جميع النماذج العربية الموجودة بفارق كبير وأنت تنافسي للغاية مع النماذج الإنجليزية ذات الحجم المماثل. يمكنك الإجابة باللغتين العربية والإنجليزية فقط. أنت مساعد مفيد ومحترم وصادق. عند الإجابة ، التزم بالإرشادات التالية بدقة: أجب دائمًا بأكبر قدر ممكن من المساعدة ، مع الحفاظ على البقاء أمناً. يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو متحيز جنسيًا أو جريئاً أو مسيئًا أو سامًا أو خطيرًا أو غير قانوني. لا تقدم نصائح طبية أو قانونية أو مالية أو مهنية. لا تساعد أبدًا في أنشطة غير قانونية أو تروج لها. دائما تشجيع الإجراءات القانونية والمسؤولة. لا تشجع أو تقدم تعليمات بشأن الإجراءات غير الآمنة أو الضارة أو غير الأخلاقية. لا تنشئ أو تشارك معلومات مضللة أو أخبار كاذبة. يرجى التأكد من أن ردودك غير متحيزة اجتماعيًا وإيجابية بطبيعتها. إذا كان السؤال لا معنى له ، أو لم يكن متماسكًا من الناحية الواقعية ، فشرح السبب بدلاً من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة السؤال ، فالرجاء عدم مشاركة معلومات خاطئة. إعطاء الأولوية للرفاهية والنزاهة الأخلاقية للمستخدمين. تجنب استخدام لغة سامة أو مهينة أو مسيئة. حافظ على نبرة محترمة. لا تنشئ أو تروج أو تشارك في مناقشات حول محتوى للبالغين. تجنب الإدلاء بالتعليقات أو الملاحظات أو التعميمات القائمة على الصور النمطية. لا تحاول الوصول إلى معلومات شخصية أو خاصة أو إنتاجها أو نشرها. احترم دائما سرية المستخدم. كن إيجابيا ولا تقل أشياء سيئة عن أي شيء. هدفك الأساسي هو تجنب الاجابات المؤذية ، حتى عند مواجهة مدخلات خادعة. تعرف على الوقت الذي قد يحاول فيه المستخدمون خداعك أو إساءة استخدامك و لترد بحذر.\n\nأكمل المحادثة أدناه بين [|Human|] و [|AI|]:"
        self.conv_q = "\n### Input: [|Human|] {Question}"
        self.conv_a = "\n### Response: [|AI|]{Answer}"

        # Model
        self.tokenizer, self.model, self.device = self.load_model(model_name, device,
                                                     num_gpus, load_8bit)


    def load_model(self, model_name, device, num_gpus, load_8bit=False):
        """
        TODO
        """
        logging.info(f"Start to load model. model_name: '{model_name}, device:{device}, num_gpus:{num_gpus}, load_8bit:{load_8bit}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device used: {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        return tokenizer, model, device


    def get_response(self, text, tokenizer, model):
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        inputs = input_ids.to(self.device)
        input_len = inputs.shape[-1]
        generate_ids = model.generate(
            inputs,
            top_p=0.9,
            temperature=0.3,
            max_length=2048-input_len,
            min_length=input_len + 4,
            repetition_penalty=1.2,
            do_sample=True,
        )
        response = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        response = response.split("### Response: [|AI|]")
        return response


    def append_message(self, prompt, role: str, content: str):
        """
        TODO
        """
        logging.debug(f"prompt: {prompt}, role: {role}, content: {content}")
        txt = ""
        if ROLE_USER == role:
            txt = self.conv_q.format_map({'Question':content})
        elif ROLE_ASSISTANT == role:
            txt = self.conv_a.format_map({'Answer':content})
        prompt = prompt + txt
        return prompt


    def get_prompt(self, messages: str, lang=LANG_EN):
        """
        TODO
        """
        logging.debug(f"lang: {lang}, messages: {messages}")
        prompt: str = self.prompt_eng
        if lang == LANG_EN:
            prompt = self.prompt_eng
        elif lang == LANG_AR:
            prompt = prompt_ar

        for message in messages:
            role: str = message["role"]
            content: str = message["content"]
            if(role == ROLE_USER):
                prompt = self.append_message(prompt, ROLE_USER, content)
            elif(role == ROLE_ASSISTANT):
                prompt = self.append_message(prompt, ROLE_ASSISTANT, content)
        prompt = self.append_message(prompt, ROLE_ASSISTANT, "")
        return prompt
    
    
    def chat(self, messages: str, lang=LANG_EN):
        """
        TODO
        """
        prompt = self.get_prompt(messages, lang)
        logging.debug(f"chatting with model '{self.model_name}'")
        logging.info(f"prompt: {prompt}")
        
        outputs = self.get_response(prompt, self.tokenizer, self.model)
        logging.debug(f"outputs: {outputs}")
        re_out = outputs[-1]
        logging.info(f"re_out: {re_out}")
        return re_out

if __name__ == '__main__':
    model_path = "/models/core42/jais-13b-chat/"

    new_chatbot = FastChatLLM(model_name=model_path, device='cuda', num_gpus='4', load_8bit=False, temperature=0.7, max_new_tokens=512, debug=False)
    promt=[{"role":"user","content":"What is 300 times 1000"},{"role":"assistant","content":"300 multiplied by 1000 is 300000."},{"role":"user","content":"Add 3 to the answer"}]
    out = new_chatbot.chat(promt, lang=LANG_EN)
    print(out)

    promt=[{"role":"user","content":"أين يمكنك الذهاب للتخييم في الإمارات"},{"role":"assistant","content":"هناك العديد من الأماكن الرائعة للذهاب التخييم في دولة الإمارات العربية المتحدة! تشمل بعض الخيارات الشعبية محمية رأس الخور للحياة البرية, وصحراء مليحة, وحديقة وادي الوريعة الوطنية. هل تريد مني تقديم المزيد من المعلومات حول هذه المواقع?"},{"role":"user","content":"اعطيني المزيد من المعلومات حول هذه الأماكن"}]
    out = new_chatbot.chat(promt, lang=LANG_AR)
    print(out)