---
authors:
- admin
bio: I was using the brain to understand math. Now I am using math to understand the brain.
email: "rklymentiev@gmail.com"
name: Ruslan Klymentiev
organizations:
- name: ""
  url: ""
role: 50% Data Scientist, 50% Neuroscientist
social:
- icon: envelope
  icon_pack: fas
  link: '#contact'
- icon: twitter
  icon_pack: fab
  link: https://twitter.com/r_klymentiev
- icon: github
  icon_pack: fab
  link: https://github.com/rklymentiev
- icon: graduation-cap
  icon_pack: fas
  link: https://scholar.google.com/citations?user=prQO0zEAAAAJ

# interests:
#   - Computational Psychiatry
#   - Bayesian Statistics
#   - Reinforcement Learning
#   - Decision-making
# 
# education:
#   courses:
#   - course: PhD in Comptational Psychiatry
#     institution: IMPRS COMP2PSYCH
#     year: 2026
#   - course: MSc in Integrative Neuroscience
#     institution: Otto von Guericke University Magdeburg
#     year: 2022
#   - course: BSc in Radioelectronic Apparatuses
#     institution: Odessa National Polytechnic University
#     year: 2014
    
superuser: true
---

<style>
.button {
  background-color: white;
  border: 2px solid red;
  color: black;
  padding: 15px 25px;
  text-align: center;
  border-radius: 14px;
  font-size: 16px;
  cursor: pointer;
  transition-duration: 0.4s;
}

.button:hover {
  background-color: red;
}
</style>

```python
class AboutMe:
    
    def __init__(self, name=None):
        self.name = name
        self.interests = []
    
    def me(self, life_credo, learning=True):
        self.life_credo = life_credo
        if learning:
            self.activity = 'learning'
        else:
            self.activity = 'traveling'
            
    def add_interest(self, interest):
        if interest not in self.interests:
            self.interests.append(interest)

    
RK = AboutMe(name='Ruslan Klymentiev')
RK.me(life_credo='Never stop learning')
RK.add_interest('computational psychiatry')
RK.add_interest('decision making')
RK.add_interest('Bayesian statistics')
```

<!--- <center>
<a class="btn" href="CV_Klymentiev.pdf" target="_blank">
# <button class="button">CV</button>
# </a>
# </center> ---> 