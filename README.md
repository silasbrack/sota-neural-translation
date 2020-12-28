# sota-neural-translation

## Introduction

## How to run

For translation and attention plots the summary.ipynb can be run independently.

## Summary of results

|            | Seq2Seq LSTM | Seq2Seq GRU w/Attn | Transformer |
|------------|--------------|--------------------|-------------|
| CE Train   | 2.19         | 1.81               | 3.28        |
| CE Val     | 2.71         | 3.20               | 3.31        |
| CE Test    | 2.84         | 3.25               | 3.287       |
| Perplexity | 17.116       | 25.774             | 26.751      |
| BLEU       | 21.7         | 29.3               | 10.5        |
| BLEU WMT14 | 0.13         | 0.21               | 2.8         |
| Parameters | 35M          | 20M                | 50M         |			

---

###### Original:
`eine gruppe von menschen steht vor einem iglu .`\
`ein mann mit kariertem hut in einer schwarzen jacke und einer schwarz-weiß gestreiften hose spielt auf einer bühne mit einem sänger und einem weiteren gitarristen im hintergrund auf einer e-gitarre .`

###### Translation:
`a group of people stands in front of an igloo .`\
`a man in a black jacket and checkered hat wearing black and white striped pants plays an electric guitar on a stage with a singer and another guitar player in the background .`

#### Short sentence

###### Seq2Seq LSTM:
`a group  of people standing in  front of a <unk> booth .`\
`a man in a black hat and black shirt plays a a with a a a a a a a a in a a in a background .`

###### Seq2Seq GRU w/ Attn:
`a group of people standing in front of a theater .`\
`a man in a plaid hat , jacket and black striped striped striped striped striped shirt , playing a guitar with a guitar with a guitar with a guitar in a treadmill .`
###### Transformer:
`a group of people are standing in front of a large building .`\
`a man in a white shirt and jeans is playing a guitar on a stage .`
