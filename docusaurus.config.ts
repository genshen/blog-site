import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Blog',
  tagline: 'Blog site of gensh.me',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://blog.gensh.me',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'genshen', // Usually your GitHub org/user name.
  projectName: 'blog', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: false, // sidebar.ts is not used.
        blog: {
          showReadingTime: true,
          routeBasePath: '/',
          postsPerPage: 3,
          blogDescription: 'A docusaurus powered blog!',
          // blogSidebarTitle: 'All posts',
          blogSidebarCount: 8, // Or "ALL"
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
            copyright: `Copyright © ${new Date().getFullYear()} gensh.me.`,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/genshen/blog-site/tree/master/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Blog',
      logo: {
        alt: 'Blog Logo',
        src: 'img/logo.svg',
      },
      items: [
        { to: '/', label: 'Blog', position: 'left' },
        { to: '/archive', label: 'Archive', position: 'left' },
        { to: '/tags', label: 'Tags', position: 'left' },
        { to: 'about', label: 'About', position: 'left' }, {
          href: 'https://github.com/genshen',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Social',
          items: [
            {
              label: 'Blog',
              to: '/',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/genshen',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} gensh.me.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'armasm', 'fortran','cmake', 'diff'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
