"""
Vision/Multimodal (Visão/Multimodal)

Técnica para analisar imagens usando LLMs com capacidade de visão como GPT-4o.
Combina entendimento de texto e imagem para análise abrangente.

Recursos:
- Descrição e análise de imagens
- Detecção e identificação de objetos
- Extração de texto de imagens (OCR)
- Interpretação de gráficos e diagramas

Requisitos:
- Modelo com capacidade de visão (gpt-4o, gpt-4o-mini)
- Imagens em formatos suportados (PNG, JPEG, GIF, WebP)

Casos de uso:
- Legendagem de imagens
- Perguntas e respostas visuais
- Análise de documentos
- Interpretação de gráficos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import base64
from langchain_core.messages import HumanMessage
from config import (
    get_llm,
    TokenUsage,
    extract_tokens_from_response,
    print_token_usage,
    print_total_usage
)

# Rastreador global de tokens
token_tracker = TokenUsage()


def codificar_imagem_base64(caminho_imagem: str) -> str:
    """
    Codifica um arquivo de imagem para string base64.

    Args:
        caminho_imagem: Caminho para o arquivo de imagem

    Retorna:
        String base64 codificada da imagem
    """
    with open(caminho_imagem, "rb") as arquivo_imagem:
        return base64.standard_b64encode(arquivo_imagem.read()).decode("utf-8")


def obter_tipo_media_imagem(caminho_imagem: str) -> str:
    """Obtém o tipo de mídia baseado na extensão do arquivo."""
    ext = Path(caminho_imagem).suffix.lower()
    tipos_media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return tipos_media.get(ext, "image/png")


def analisar_imagem(caminho_imagem: str, prompt: str = "Descreva esta imagem em detalhes.") -> str:
    """
    Analisa uma imagem usando um LLM com capacidade de visão.

    Args:
        caminho_imagem: Caminho para o arquivo de imagem
        prompt: Pergunta ou instrução sobre a imagem

    Retorna:
        Análise do LLM sobre a imagem
    """
    llm = get_llm(temperature=0.3)

    # Codifica imagem
    base64_imagem = codificar_imagem_base64(caminho_imagem)
    tipo_media = obter_tipo_media_imagem(caminho_imagem)

    # Cria mensagem com imagem
    mensagem = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{tipo_media};base64,{base64_imagem}"
                }
            }
        ]
    )

    response = llm.invoke([mensagem])

    # Extrai e registra tokens
    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def descrever_imagem(caminho_imagem: str) -> str:
    """Gera uma descrição detalhada de uma imagem."""
    return analisar_imagem(
        caminho_imagem,
        "Descreva esta imagem em detalhes. Inclua informações sobre objetos, cores, composição e quaisquer características notáveis."
    )


def extrair_texto_da_imagem(caminho_imagem: str) -> str:
    """Extrai texto visível em uma imagem (funcionalidade tipo OCR)."""
    return analisar_imagem(
        caminho_imagem,
        "Extraia e liste todo o texto visível nesta imagem. Inclua quaisquer rótulos, legendas ou conteúdo escrito."
    )


def analisar_grafico(caminho_imagem: str) -> str:
    """Analisa uma imagem de gráfico."""
    return analisar_imagem(
        caminho_imagem,
        """Analise este gráfico. Forneça:
1. Tipo de gráfico (barras, linhas, pizza, etc.)
2. Que dados ele representa
3. Principais insights ou tendências
4. Quaisquer valores ou outliers notáveis"""
    )


def identificar_objetos(caminho_imagem: str) -> str:
    """Identifica e lista objetos em uma imagem."""
    return analisar_imagem(
        caminho_imagem,
        "Identifique e liste todos os objetos, pessoas ou elementos distintos visíveis nesta imagem. Forneça uma breve descrição de cada um."
    )


def responder_sobre_imagem(caminho_imagem: str, pergunta: str) -> str:
    """Responde uma pergunta específica sobre uma imagem."""
    return analisar_imagem(caminho_imagem, pergunta)


def comparar_imagens(caminho_imagem1: str, caminho_imagem2: str, aspecto: str = "geral") -> str:
    """
    Compara duas imagens.

    Args:
        caminho_imagem1: Caminho para a primeira imagem
        caminho_imagem2: Caminho para a segunda imagem
        aspecto: Que aspecto comparar (geral, cores, objetos, etc.)

    Retorna:
        Análise de comparação
    """
    llm = get_llm(temperature=0.3)

    # Codifica ambas as imagens
    base64_imagem1 = codificar_imagem_base64(caminho_imagem1)
    base64_imagem2 = codificar_imagem_base64(caminho_imagem2)
    tipo_media1 = obter_tipo_media_imagem(caminho_imagem1)
    tipo_media2 = obter_tipo_media_imagem(caminho_imagem2)

    prompt = f"""Compare estas duas imagens com foco em {aspecto}.
Descreva:
1. Similaridades entre as imagens
2. Diferenças entre as imagens
3. Avaliação geral"""

    mensagem = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{tipo_media1};base64,{base64_imagem1}"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{tipo_media2};base64,{base64_imagem2}"}
            }
        ]
    )

    response = llm.invoke([mensagem])

    input_tokens, output_tokens = extract_tokens_from_response(response)
    token_tracker.add(input_tokens, output_tokens)
    print_token_usage(input_tokens, output_tokens)

    return response.content


def analisar_diagrama(caminho_imagem: str) -> str:
    """Analisa um diagrama técnico ou fluxograma."""
    return analisar_imagem(
        caminho_imagem,
        """Analise este diagrama ou fluxograma. Descreva:
1. O tipo de diagrama
2. Os componentes ou elementos principais
3. As relações ou fluxo entre os elementos
4. O propósito ou mensagem geral do diagrama"""
    )


def main():
    print("=" * 60)
    print("VISION/MULTIMODAL (VISÃO/MULTIMODAL) - Demo")
    print("=" * 60)

    token_tracker.reset()

    # Caminho para imagens de exemplo
    diretorio_exemplos = Path(__file__).parent.parent.parent / "sample_data" / "images"

    if not diretorio_exemplos.exists():
        print(f"\nErro: Diretório de imagens de exemplo não encontrado em {diretorio_exemplos}")
        print("Por favor, certifique-se que o diretório sample_data/images existe com imagens de exemplo.")
        return

    # Verifica imagens disponíveis
    imagens_disponiveis = list(diretorio_exemplos.glob("*.*"))
    if not imagens_disponiveis:
        print(f"\nNenhuma imagem encontrada em {diretorio_exemplos}")
        return

    print(f"\n📷 Encontrada(s) {len(imagens_disponiveis)} imagem(ns):")
    for img in imagens_disponiveis:
        print(f"   - {img.name}")

    # Exemplo 1: Análise de Gráfico
    caminho_grafico = diretorio_exemplos / "chart.png"
    if caminho_grafico.exists():
        print("\n\n📊 ANÁLISE DE GRÁFICO")
        print("-" * 40)
        print(f"\nAnalisando: {caminho_grafico.name}")
        resultado = analisar_grafico(str(caminho_grafico))
        print(f"\n📋 Análise:\n{resultado}")

    # Exemplo 2: Análise de Diagrama
    caminho_diagrama = diretorio_exemplos / "diagram.png"
    if caminho_diagrama.exists():
        print("\n\n📐 ANÁLISE DE DIAGRAMA")
        print("-" * 40)
        print(f"\nAnalisando: {caminho_diagrama.name}")
        resultado = analisar_diagrama(str(caminho_diagrama))
        print(f"\n📋 Análise:\n{resultado}")

    # Exemplo 3: Descrição de Foto
    caminho_foto = diretorio_exemplos / "photo.jpg"
    if caminho_foto.exists():
        print("\n\n🖼️ DESCRIÇÃO DE FOTO")
        print("-" * 40)
        print(f"\nAnalisando: {caminho_foto.name}")
        resultado = descrever_imagem(str(caminho_foto))
        print(f"\n📋 Descrição:\n{resultado}")

    # Exemplo 4: Identificação de Objetos
    if caminho_foto.exists():
        print("\n\n🔍 IDENTIFICAÇÃO DE OBJETOS")
        print("-" * 40)
        print(f"\nIdentificando objetos em: {caminho_foto.name}")
        resultado = identificar_objetos(str(caminho_foto))
        print(f"\n📋 Objetos Encontrados:\n{resultado}")

    # Exemplo 5: Perguntas e Respostas Visuais
    if caminho_grafico.exists():
        print("\n\n❓ P&R VISUAL")
        print("-" * 40)
        pergunta = "Qual é o maior valor mostrado neste gráfico?"
        print(f"\nImagem: {caminho_grafico.name}")
        print(f"Pergunta: {pergunta}")
        resultado = responder_sobre_imagem(str(caminho_grafico), pergunta)
        print(f"\n📋 Resposta:\n{resultado}")

    # Exemplo 6: Comparação de Imagens
    if caminho_grafico.exists() and caminho_diagrama.exists():
        print("\n\n🔄 COMPARAÇÃO DE IMAGENS")
        print("-" * 40)
        print(f"\nComparando: {caminho_grafico.name} vs {caminho_diagrama.name}")
        resultado = comparar_imagens(str(caminho_grafico), str(caminho_diagrama), "tipo de visualização e propósito")
        print(f"\n📋 Comparação:\n{resultado}")

    print_total_usage(token_tracker, "TOTAL - Vision/Multimodal")

    print("\n\n" + "=" * 60)
    print("Nota: Capacidades de visão requerem um modelo habilitado")
    print("para visão como gpt-4o ou gpt-4o-mini. Certifique-se que")
    print("OPENAI_MODEL está configurado corretamente.")
    print("=" * 60)

    print("\nFim do demo Vision/Multimodal")
    print("=" * 60)


if __name__ == "__main__":
    main()
