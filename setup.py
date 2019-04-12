from distutils.core import setup
setup(
  name = 'feather_nlp',         # How you named your package folder (MyLib)
  packages = ['feather_nlp'],   # Chose the same as "name"
  version = '0.1.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'This library works on top of RASA-NLU to make its nlu functionalities \navailable in low powered devices like raspberry pi',   # Give a short description about your library
  author = 'Chidhambararajan',                   # Type in your name
  author_email = 'chidha1434@protonmail.com',      # Type in your E-Mail
  url = 'https://github.com/Chidhambararajan/feather_nlp',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['NLP', 'machine learning', 'rasa nlu'],   # Keywords that define your package best
  install_requires=[
          'rasa_nlu',
          'fuzzywuzzy[speedup]'
      ],
  classifiers=[
    'Development Status :: 4 - Alpha ',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
