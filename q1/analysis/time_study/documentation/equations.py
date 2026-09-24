"""Native OMML formulas with explicit fractions, indices and delimiters."""
from docx.oxml import OxmlElement as E
from docx.oxml.ns import qn


def run(text,plain=False):
    x=E("m:r")
    if plain:
        p=E("m:rPr");n=E("m:nor");n.set(qn("m:val"),"1");p.append(n);x.append(p)
    t=E("m:t");t.text=text;x.append(t);return x


def fill(parent,values):
    if not isinstance(values,(list,tuple)):values=[values]
    for v in values:parent.append(run(v) if isinstance(v,str) else v)
    return parent


def sub(base,index):
    x=E("m:sSub");x.append(fill(E("m:e"),base));x.append(fill(E("m:sub"),index));return x


def sup(base,index):
    x=E("m:sSup");x.append(fill(E("m:e"),base));x.append(fill(E("m:sup"),index));return x


def ss(base,lower,upper):
    x=E("m:sSubSup");x.append(fill(E("m:e"),base));x.append(fill(E("m:sub"),lower));x.append(fill(E("m:sup"),upper));return x


def frac(num,den):
    x=E("m:f");x.append(fill(E("m:num"),num));x.append(fill(E("m:den"),den));return x


def delim(values,left="(",right=")"):
    x=E("m:d");p=E("m:dPr")
    for tag,val in [("begChr",left),("endChr",right)]:
        e=E("m:"+tag);e.set(qn("m:val"),val);p.append(e)
    x.append(p);x.append(fill(E("m:e"),values));return x


def sqrt(values):
    x=E("m:rad");p=E("m:radPr");h=E("m:degHide");h.set(qn("m:val"),"1");p.append(h);x.append(p);x.append(E("m:deg"));x.append(fill(E("m:e"),values));return x


def nary(lower,values,char="∑"):
    x=E("m:nary");p=E("m:naryPr")
    for tag,val in [("chr",char),("limLoc","subSup"),("supHide","1")]:
        e=E("m:"+tag);e.set(qn("m:val"),val);p.append(e)
    x.append(p);x.append(fill(E("m:sub"),lower));x.append(E("m:sup"));x.append(fill(E("m:e"),values));return x


def formula(number):
    x=lambda b,i:sub(b,i)
    if number==1:
        values=[ss("x","i","T"),"=",frac("1",delim(x("B","i"),"|","|")),nary(["j∈",x("B","i")],x("h","j")),",  ",ss("x","i","T"),"∈",sup("ℝ","768")]
    elif number==2:
        values=[ss("x","i","A"),"=",delim([ss("μ","i","e"),";",ss("μ","i","l"),";",ss("σ","i","l")],"[","]"),"∈",sup("ℝ","818")]
    elif number==3:
        values=[ss("x","i","V"),"=",delim([ss("μ","i","v"),";",ss("σ","i","v")],"[","]"),"∈",sup("ℝ","56")]
    elif number==4:
        values=[ss("τ","n","A"),"=",sup("δ","A"),"+",frac("n","16000"),",  ",ss("τ","j","V"),"=",x(run("PTS",True),"j"),"−",x("t","0")]
    elif number==5:
        values=[ss("r","j",delim("m")),"=",delim([run("sample_id",True),", m, ",run("original_index",True),", ",delim([x("a","j"),",",x("b","j")],"[",")"),", ",x(run("valid",True),"j"),", ",x("q","j")])]
    elif number==6:
        values=[x("ℓ","ij"),"=",run("max",True),delim(["0, ",run("min",True),delim([x("e","i"),",",x("b","j")]),"−",run("max",True),delim([x("s","i"),",",x("a","j")])])]
    elif number==7:
        values=[x("w","ij"),"=",frac([x("ℓ","ij"),x("q","j")],nary("r",[x("ℓ","ir"),x("q","r")]))]
    elif number==8:
        values=[x("μ","i"),"=",nary("j",[x("w","ij"),x("x","j")]),",  ",x("σ","i"),"=",sqrt(nary("j",[x("w","ij"),sup(delim([x("x","j"),"−",x("μ","i")]),"2")]))]
    elif number==9:
        values=[x(run("coverage",True),"i"),"=",frac(delim([x("I","i"),"∩",x("U","G")],"|","|"),delim(x("I","i"),"|","|"))]
    elif number==10:
        values=[ss("h","j",delim("m")),"=",run("GELU",True),delim([run("LayerNorm",True),delim([x("W","m"),ss("x","j",delim("m")),"+",x("b","m")])])]
    elif number==11:
        values=[x(run("score",True),"ij"),"=",frac([x("Q","i"),ss("K","j","T")],sqrt("16")),"−0.5",sup(delim(frac(x("Δt","ij"),"0.25")),"2"),"+",run("log",True),delim(x("q","j"))]
    elif number==12:
        values=[x("α","ij"),"=",x(run("softmax",True),"j"),delim(x(run("score",True),"ij")),",  ",delim(x("Δt","ij"),"|","|"),"≤0.5 ",run("s",True)]
    else:raise ValueError(number)
    return fill(E("m:oMath"),values)
