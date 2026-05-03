class LlmSpeed < Formula
  include Language::Python::Virtualenv

  desc "Benchmark any LLM on any hardware. CLI for the llm-speed.com flywheel."
  homepage "https://llm-speed.com"
  url "https://files.pythonhosted.org/packages/source/l/llm-speed/llm-speed-0.0.1.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "Apache-2.0"

  depends_on "rust" => :build # for cryptography wheels on some platforms
  depends_on "python@3.12"

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/source/a/anyio/anyio-4.6.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/source/c/certifi/certifi-2024.8.30.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "cffi" do
    url "https://files.pythonhosted.org/packages/source/c/cffi/cffi-1.17.1.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "cryptography" do
    url "https://files.pythonhosted.org/packages/source/c/cryptography/cryptography-43.0.1.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/source/h/h11/h11-0.14.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "httpcore" do
    url "https://files.pythonhosted.org/packages/source/h/httpcore/httpcore-1.0.5.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/source/h/httpx/httpx-0.27.2.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/source/i/idna/idna-3.10.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/source/m/markdown-it-py/markdown-it-py-3.0.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/source/m/mdurl/mdurl-0.1.2.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "psutil" do
    url "https://files.pythonhosted.org/packages/source/p/psutil/psutil-6.1.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "pycparser" do
    url "https://files.pythonhosted.org/packages/source/p/pycparser/pycparser-2.22.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "pygments" do
    url "https://files.pythonhosted.org/packages/source/p/pygments/pygments-2.18.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-13.9.2.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "sniffio" do
    url "https://files.pythonhosted.org/packages/source/s/sniffio/sniffio-1.3.1.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "llm-speed", shell_output("#{bin}/llm-speed --version")
  end
end
