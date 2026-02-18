# Pissa-Without-Regret
Reproducing Pissa from scratch (no HF) and applying findings of the Lora-Without-Regret blog.


## Pissa-Without-Regret findings

This an extension from previous work of reporudicing the Lora-Without-Regret blog and investigating LoRA and FullFT optimal LR ratio.

### Let' see how different is Pissa using the blog's recommended configuration.

We use similar setup to our Lora-without-regret one using:
- `A` initalized using uniform distributin and `B` is zero
- We use a constant `alpha` value of `32` and factor by `1/r`
- We set a fixed `lr` (no scheduler) used by both adapters
- We train  `Distil-bert-uncased` on a `10k` subset of AG-News (classifcation)

   <br>
  
<p align="center">
  <img src="assets/pissa" width="700"/>
</p>
<br>

Here is the FullFT plot:


