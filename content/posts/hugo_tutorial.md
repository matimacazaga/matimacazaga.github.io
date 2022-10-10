---
title: "How to create your own blog using Github Pages"
date: 2022-10-10
# weight: 1
# aliases: ["/first"]
tags: ["hugo", "GitHub Pages"]
author: "Matias Macazaga"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Step-by-step tutorial for creating a personal blog using Hugo and Github Pages."
canonicalURL: "https://canonical.url/to/page"
# disableHLJS: true # to disable highlightjs
disableShare: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "/img/blogging.png" # image path/url
    alt: "Blogging" # alt text
    caption: "" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
# editPost:
#     URL: "https://github.com/<path_to_repo>/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: true # to append file path to Edit link
---

There are many good reasons to have a personal blog or website. For instance, it is an excellent way of showcasing your skills and sharing knowledge with your partners.

Tools like Wordpress and Wix allow you to easily create your first blog. However, they can be costly to maintain and configure. For this reason, I will show you in this post the same process I followed to create and deploy this website for free using [Hugo](https://gohugo.io/) and [Github Pages](https://pages.github.com/).

## What is Hugo?

{{<figure src="/img/hugo-logo-wide.png" alt="HUGO" position="center" style="border-radius: 8px;" caption="HUGO Web Framework">}}

Hugo is an open-source static site generator written in [`Go`](https://go.dev/). It is used to build content-focussed websites in a flexible and fast manner (if you don't believe me, just watch [this](https://www.youtube.com/watch?v=w7Ft2ymGmfc) YouTube tutorial). Although Hugo sites are highly customizable, you do not need any advanced programming skill to work with it.

Hugo has support for [Markdown](https://www.markdownguide.org/), an easy-to-use markup language that makes your life easier for writing content. However, the magic does not end here. Hugo comes with built-in *shortcodes*, which are code snippets you can utilize inside your Markdown content files for using custom templates. Hugo will render the *shortcode* using a predefined template, circumventing the need of using raw `HTML` code. An example of this is the rounded image of the Hugo logo shown above that contains a nicely displayed caption.

The last feature I would like to highlight is that Hugo has supports for themes. You can choose one from the [official list](https://themes.gohugo.io/) and start adding your content with little effort.

## What is GitHub Pages?

{{<figure src="/img/github-logo.png" alt="HUGO" position="center" style="border-radius: 8px; background: white" caption="GitHub - [source](https://1000logos.net/github-logo/)">}}

GitHub pages are public web pages for users and organizations that are freely hosted on Github's `github.io` domain or on a custom domain name. GitHub Pages allows to create an entire website directly from a repository on GitHub.com.

GitHub Pages basically takes HTML, CSS and JS files from a repository, runs them through a build process, and publishes the website.

## Creating your blog

### Prerequisites

There are some prerequisites you need to fulfill before start working with Hugo and GithubPages:

- A [GitHub Account](https://github.com/join) for creating the site's repository.
- [Git](https://git-scm.com/) for managing the project.
- Some familiarity with [Markdown](https://www.markdownguide.org/). Don't worry, Markdown is simple and easy to learn.

### Installing Hugo

Hugo supports multiple platforms. Below you can find quick installation instructions depending on your OS.

#### macOs and Linux
If you are on macOS or Linux, you can install Hugo with the following one-liner thanks to [Homebrew](https://brew.sh/):

```bash
brew install hugo
```

#### Windows

On Windows, you need to install [Chocolatey](https://chocolatey.org/) first and then type the following on the PowerShell:

```powershell
choco install hugo-extended -confirm
```

You can check Hugo's installation by using

```bash
hugo version
```

### Create a GitHub repository

The next step is creating a GitHub repository. Go to [this](github.com/new/) and login if necessary. Name the new repository *\<username\>.github.io* so your website is published at *https://\<username\>.github.io*. Set the visibility to *Public* and initialize the repository without a `README` and MIT License.

Once the repository is created, clone it to your local computer and open it in VS Code (or your IDE of preference).

### Create a Hugo Project

For creating a new Hugo project, open the terminal in the same folder of your repository, and type the following

```bash
hugo new site ./ --force
```

The `--force` argument is used because the folder we are using is not empty (we already have the license in there).

By default, Hugo uses `TOML` for configuration, but you can change to `YAML` by adding `-f yml` to the command above. The format of the configuration file will depend on the theme you choose (More on this below).

You will see that several new folders will be created. Now, we are ready for installing one of the gorgeous themes available.

### Installing a theme

As mentioned above, you can use a theme from the [Hugo library](https://themes.gohugo.io/). The initial configuration will vary from one theme to another, but fortunately most of them have a very good documentation page.

This time, I will be showing how to setup the [PaperMod theme](https://adityatelange.github.io/hugo-PaperMod/) (the one I'm currently using). From the [official documentation](https://adityatelange.github.io/hugo-PaperMod/posts/papermod/papermod-installation/), we can see that there are several methods for installing/updating the theme. I used the second method, but you can choose what best suits your needs.

```bash
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive # needed when you reclone your repo (submodules may not get cloned automatically)

```

The theme files will be included in the `theme` folder. Next step is to change the `config.yml` file. From the same documentation page, we can copy and paste the required settings. Additionally, you can find a description of the main features of the theme in [this post](https://adityatelange.github.io/hugo-PaperMod/posts/papermod/papermod-features/) and detailed information of the available variables [here](https://adityatelange.github.io/hugo-PaperMod/posts/papermod/papermod-variables/).

### Running your website locally

For running your website locally and checking that the theme is working, use:

```bash
hugo server
```

and open *https://localhost:1313/* in your web browser. A handy feature of Hugo is that whenever you update something in your blog, it will be reflected on the site automatically as long as the hugo server process is running.

For adding your first blog post, you need to create a `post.md` file in the `archetypes` folder with the following content, which will serve as a starting base for all your blog posts.

```markdown
---
title: "My 1st post"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["first"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Desc Text."
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
```

Then, create a new folder called `posts` under the `content` folder. You can now automatically create posts using the following hugo command:

```bash
hugo new --kind post /content/posts/<name>
```

This creates a new file in the `content/posts` folder that can be used as base for your blog post. Put some text in there and hit save to see the changes on the site.

### Generate and publish the website

You need to make some extra changes before being ready for publishing your website. In the `config.yml`, modified the `baseurl` parameter using your website name:

```yaml
baseurl = "https://username.github.io"
```

After doing that, create a `gh-pages.yml` file under `.github/workflows` folder (create the folder and sub-folder if necessary).

> NOTE: make sure to include the "." before github in the `.github/workflows` folder. 

Copy the following lines inside the `gh-pages.yml` file:

```yaml
name: Deploy Hugo to Pages

on:
  push:
    paths-ignore:
      - "images/**"
      - "LICENSE"
      - "README.md"
    branches:
      - main
  workflow_dispatch:
    # manual run
    inputs:
      hugoVersion:
        description: "Hugo Version"
        required: false
        default: "0.83.0"

# Allow one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: true

# Default to bash
defaults:
  run:
    shell: bash

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: "0.83.0"
    steps:
      - name: Check version
        if: ${{ github.event.inputs.hugoVersion }}
        run: export HUGO_VERSION="${{ github.event.inputs.hugoVersion }}"
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_${HUGO_VERSION}_Linux-64bit.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb
      - name: Checkout
        uses: actions/checkout@v3
        # with:
        #   ref: exampleSite
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v1
      - name: Get Theme
        run: git submodule update --init --recursive
      - name: Update theme to Latest commit
        run: git submodule update --remote --merge
      - name: Build with Hugo
        run: |
          hugo \
            --buildDrafts --gc --verbose \
            --baseURL ${{ steps.pages.outputs.base_url }}
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v1
        with:
          path: ./public
  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v1
```

Next, we are ready to generate the website by typing `hugo` in the command line. All your website files will be placed on the `public` folder. Commit and push the changes to the GitHub repository.

Finally, go to your repository page and navigate to *Settings -> Pages* and under the *Source* option select *GitHub Actions*. Wait a few seconds until your site is deployed (you can check the deployment progress in the *Actions* tab) and visit it to check everything is correctly displayed.

### Conclusion

This is how you can create a personal website using Hugo and GitHub Pages. Best of all, after you went through all the initial setup, all you have to worry about is finding content for writing your new posts. Reach me out if you have any doubts or comments. Happy blogging!

## Main Resources

- [Hugo and Github Pages](https://4bes.nl/2021/08/29/create-a-website-with-hugo-and-github-pages/)
- [PaperMod Theme by *adityatelange*](https://adityatelange.github.io/hugo-PaperMod/) 
- [Host Hugo on GitHub](https://gohugo.io/hosting-and-deployment/hosting-on-github/)

