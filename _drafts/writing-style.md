---
layout:      post
title:       "Writing Style"
tags:        [general]
authors:
    - name: First Author
      link: https://link/to/personal/webpage
    - name: Second Author
      link: https://link/to/personal/webpage
    - name: Last Author
      link: https://link/to/personal/webpage
comments:   true
image:      /assets/images/trinity.jpg
---

The title is shown automatically.
You do not need to write `# Title`.

## Sections and Subsections

You can make a section by writing `## Section`.
Subsections use triple hashes: `### Subsection`.

### Example of Subsection

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Equations

You can render inline math by using double dollar signs.
For example, `$$f(x) = x^2$$` renders as $$f(x) = x^2$$.
Display math can be rendered by writing a LaTeX environment.
For example,

```
\begin{equation} \label{eq:f}
    f(x) = x^2
\end{equation}
```

renders as

\begin{equation} \label{eq:f}
    f(x) = x^2.
\end{equation}

You can then reference this equation with `\eqref{eq:f}`: \eqref{eq:f}.

## Code

You can render inline code by using single backticks.
For example, `` `code` `` generates `code`.
A code pane can be rendered by using triple backticks. 

For example,

````
```
def f(x):
    return x ** 2
```
````

renders as

```
def f(x):
    return x ** 2
```

You can add syntax highlighting by writing the name of the programming language right after the first triple backticks:

````
```python
def f(x):
    return x ** 2
```
````

```python
def f(x):
    return x ** 2
```

## Tables

An example of a table is as follows:

```
{: #table-test }
| Head 1  | Head 2  |
| :------ | :------ |
| Value 1 | Value 2 |

{% raw %}{% include table_caption.html 
    name="Table 1"
    caption="Bla"
%}{% endraw %}
```

{: #table-test }
| Head 1  | Head 2  |
| :------ | :------ |
| Value 1 | Value 2 |

{% include table_caption.html 
    name="Table 1"
    caption="This is a caption"
%}

You can then reference this table with `[Table 1](#table-test)`: [Table 1](#table-test).
Note that you must number the tables yourself.
References to tables should capitalised and not abbreviated.

## Images

An example of an image is as follows:

```
{% raw %}{% include image.html
    name="Figure 1"
    ref="draft"
    alt="Draft"
    url="https://lh3.googleusercontent.com/proxy/AqPURYdtoNJirZJ9mUqtVZ2ki7UTr1X3GHQTg5jHynsPgEYYLmlC9MzREAKarm8nTi7MFvFb3_DNAABSHyelGaYXmqZr1nc4KeB2o5CT_A-xj1bCjA9LfzZm"
    width=300
%}{% endraw %}
```

{% include image.html
    name="Figure 1"
    ref="draft"
    alt="Draft"
    url="https://lh3.googleusercontent.com/proxy/AqPURYdtoNJirZJ9mUqtVZ2ki7UTr1X3GHQTg5jHynsPgEYYLmlC9MzREAKarm8nTi7MFvFb3_DNAABSHyelGaYXmqZr1nc4KeB2o5CT_A-xj1bCjA9LfzZm"
    width=500
%}

You can then reference this table with `[Figure 1](#figure-draft)`: [Figure 1](#figure-draft).
Note that you must number the figures yourself.
Moreover, note that the link to the figure is `figure-` plus the value given to `ref` in the include.
References to figures should capitalised and not abbreviated.

If you wish to show a file instead of an URL, then you should set `src` instead of `url` in the include:

```
{% raw %}{% include image.html
    name="Figure 2"
    ref="trinity"
    alt="Trinity"
    src="trinity.jpg"
    width=500
%} {% endraw %}
```

{% include image.html
    name="Figure 2"
    ref="trinity"
    alt="Trinity"
    src="trinity.jpg"
    width=500
%}

Setting `src="trinity.jpg"` searches for the image `assets/images/trinity.jpg`.
Please collect images you upload in a folder `assets/image/name-of-post` named descriptively.

## Lists

Lists are sentences formatted in a structured way and as should be punctuated accordingly. 
For example, these are all vehicles:

* car,
* airplane, and
* bike.

If the items of a list are parts of sentences, use semicolons rather than commas to separate the list items.
For example,

* this could be something, written in some way;
* this could be something else, written in another way; and
* this is the last item in the list, wrapping up.

## References

References should be formatted as follows:

* if there is one author: [Lastname (2020)](https://link/to/paper);

* if there are two authors: [Lastname1 & Lastname2 (2020)](https://link/to/paper); and

* if there are thee or more authors: [Lastname1 et al. (2020)](https://link/to/paper).

References should link to a PDF of the paper, if possible, or otherwise a page where the paper can be found.
