pkgs = ['docx', 'pptx', 'matplotlib', 'reportlab', 'pandas', 'torch', 'gymnasium', 'seaborn', 'PIL', 'scipy', 'sklearn']
for pkg in pkgs:
    try:
        __import__(pkg)
        print(f"{pkg}: available")
    except ImportError:
        print(f"{pkg}: NOT available")
