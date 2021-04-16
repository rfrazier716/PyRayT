from docutils import nodes

encyclopedia_icon_classes = "fas fa-book rp-encyclopedia"

def setup(app):
    app.add_role('rp', rp_link)
    app.add_role('encyclopedia',generic_url)

def rp_link(name, rawtext, text, lineno, inliner, options={}, content=[]):
    if text == "_":
        node = nodes.raw('',f'<i class="{encyclopedia_icon_classes}""></i>',format='html',**options)
        return [node], []

    else:
        url = "https://www.rp-photonics.com/" + text +".html"
        node = nodes.raw('',f'<a href="{url}"><i class="{encyclopedia_icon_classes}""></i></a>',format='html',**options)
        return [node], []

def generic_url(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.raw('',f'<a href="{text}"><i class="{encyclopedia_icon_classes}""></i></a>',format='html',**options)
    return [node], []
    

    
