from typing import Dict, Sequence
import transformers
from modeling_phi import PhiForCausalLM
from tokenization_codegen import CodeGenTokenizer

prompt2 = """Question: A simple machine that helps move a flag up a flagpole is
Answer: a pulley

Question: An example of a learned behavior is a
Answer: dog laying down on command.

Question: During an experiment, a student reports that a liquid turned green when mixed with another liquid. This is an example of
Answer: an observation

Question: Which is an example of a chemical change?
Answer: a nail rusting in water

Question: Which is an example of a chemical change?
Answer: a rusting car fender

Question: Which device converts kinetic energy into electrical energy?
Answer: generator

Question: If a class wants to measure the speed of a bicycle during an outdoor lab exercise, which two devices would be most useful?
Answer: stopwatch and tape measure

Question: Scientists use the term "light year" to describe
Answer: the distance light travels in one year.

Question: Two scientists doing the same experiment got different results. Which would be the best way to figure out which result might be correct?
Answer: More scientists should do the experiment.

Question: Which characteristic is inherited rather than learned?
Answer: having blue eyes

Question: A caterpillar changing into a butterfly is an example of
Answer: metamorphosis.

Question: Which technological advancement has done the most to improve the accuracy of weather predictions?
Answer: satellites

Question: Which of these is an inherited behavior?
Answer: a spider spinning a web

Question: Which of these can be recycled in an attempt to conserve resources?
Answer: aluminum

Question: Which of these is a negative effect associated with some types of new technology?
Answer: decreased physical activity

Question: Which class of elements best conducts electricity?
Answer: metals

Question: An element is identified by
Answer: its number of protons.

Question: An example of a learned behavior is
Answer: driving a car

Question: Which statement is a MAJOR mistake in the experimental design?
Answer: A control group was not used.

Question: A device that converts light energy into electricity used in many handheld calculators is a
Answer: photovoltaic cell.

Question: Which tool would a student use to measure wind speed?
Answer: anemometer

Question: The correct procedure after completing a laboratory experiment is to
Answer: collect and store recyclable material.

Question: Which ability is the most useful for making observations?
Answer: senses

Question: Which object is powered by an electrical circuit?
Answer: a flashlight

Question: What is the best way for a student to describe the results of an experiment?
Answer: in a written report

Question: An example of a renewable resource is
Answer: wood.

Question: Which of the following is an example of genetic engineering?
Answer: Inserting a gene into plants that makes them resistant to insects.

Question: A student uses a high-efficiency gasoline lawnmower to cut grass. The term "high efficiency" is used to indicate that the lawnmower
Answer: uses less energy than other lawnmowers.

Question: An important rule for students to know when heating a test tube is to
Answer: point the mouth of the test tube away from others.

Question: Which of the following is an example of an assistive device?
Answer: contact lens

Question: Which is true of scientific discoveries?
Answer: Sometimes scientific discoveries are made by accident, like the discovery of penicillin.

Question: Which object used in an experiment can safely be recycled?
Answer: an aluminum can

Question: Which of these is unique to the process of scientific investigation?
Answer: collecting data in an experiment

Question: Which is the best example of recycling?
Answer: using aluminum cans to make new products

Question: Which is an example of a physical change?
Answer: ice melting

Question: Which invention made mass production possible?
Answer: the assembly line

Question: Which is the first step in a design process?
Answer: Describe the problem.

Question: Which of the following is the best example of a custom-made product?
Answer: artificial leg

Question: A push or a pull on an object is an example of
Answer: force.

Question: What is the first step in designing a product?
Answer: identify the need or want

Question: Improvements in farming technology would most likely
Answer: increase the amount of food produced.

Question: Which did Thomas Edison invent?
Answer: light bulb

Question: Which best describes transportation technology?
Answer: a system that is used to move people and products

Question: Which technology was developed most recently?
Answer:"""
prompt1 = """Question: An important rule for students to know when heating a test tube is to
Candidate answers: (A) place a cork in the mouth of the test tube. (B) point the mouth of the test tube away from others. (C) hold the test tube loosely with the fingertips. (D) shake the test tube forcefully to keep contents mixed.
Gold answer: point the mouth of the test tube away from others.

Question: Which of the following is an example of an assistive device?
Candidate answers: (A) contact lens (B) motorcycle (C) raincoat (D) coffee pot
Gold answer: contact lens

Question: Which is true of scientific discoveries?
Candidate answers: (A) All scientific discoveries are based solely on observation and never experimentation. (B) Sometimes scientific discoveries are made by accident, like the discovery of penicillin. (C) If a scientist cannot provide the exact time and place his or her discovery was made, it is dismissed. (D) All scientific discoveries are regulated by government agencies, like the Food and Drug Administration.
Gold answer: Sometimes scientific discoveries are made by accident, like the discovery of penicillin.

Question: Which object used in an experiment can safely be recycled?
Candidate answers: (A) an aluminum can (B) a wet paper towel (C) salt spilled onto a tabletop (D) a broken graduated cylinder
Gold answer: an aluminum can

Question: Which of these is unique to the process of scientific investigation?
Candidate answers: (A) observing an event as it occurs (B) discussing results with other experts (C) publishing the results on a webpage (D) collecting data in an experiment
Gold answer: collecting data in an experiment

Question: Which is the best example of recycling?
Candidate answers: (A) using low energy appliances (B) washing and reusing plastic cups (C) using empty milk cartons as flower planters (D) using aluminum cans to make new products
Gold answer: using aluminum cans to make new products

Question: Which is an example of a physical change?
Candidate answers: (A) ice melting (B) nail rusting (C) bread baking (D) wood burning
Gold answer: ice melting

Question: Which invention made mass production possible?
Candidate answers: (A) the assembly line (B) the airplane (C) the personal computer (D) the telephone
Gold answer: the assembly line

Question: Which is the first step in a design process?
Candidate answers: (A) Revise the solution. (B) Describe the problem. (C) Test the possible solutions. (D) Identify possible solutions.
Gold answer: Describe the problem.

Question: Which of the following is the best example of a custom-made product?
Candidate answers: (A) graphing calculator (B) light bulb (C) needle nose pliers (D) artificial leg
Gold answer: artificial leg

Question: A push or a pull on an object is an example of
Candidate answers: (A) force. (B) weight. (C) energy. (D) work.
Gold answer: force.

Question: What is the first step in designing a product?
Candidate answers: (A) model a solution (B) communicate the solution (C) identify the need or want (D) build a prototype
Gold answer: identify the need or want

Question: Improvements in farming technology would most likely
Candidate answers: (A) increase the amount of food produced. (B) change global climate conditions. (C) promote unhealthy dietary choices. (D) decrease the amount of daily exercise.
Gold answer: increase the amount of food produced.

Question: Which did Thomas Edison invent?
Candidate answers: (A) microscope (B) compass (C) light bulb (D) steam engine
Gold answer: light bulb

Question: Which best describes transportation technology?
Candidate answers: (A) a system that is used to move people and products (B) an enterprise that changes raw materials into goods (C) the building and finishing of structures (D) the conversion of mechanical energy into heat energy
Gold answer: a system that is used to move people and products

Question: Which technology was developed most recently?
Candidate answers: (A) cellular telephone (B) television (C) refrigerator (D) airplane
Gold answer:"""


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=1024,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def get_model(
        base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    # replace_with_chunkllama(pretraining_length=4096)
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True,
    #                                              torch_dtype=torch.bfloat16)
    tokenizer = CodeGenTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = PhiForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model


model = PhiForCausalLM.from_pretrained("models/phi-1_5").to('cuda')
model.eval()
