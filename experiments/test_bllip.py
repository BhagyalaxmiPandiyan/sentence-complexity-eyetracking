"""Test BLLIP parser speed and probability output."""
import time

try:
    import bllipparser
    from bllipparser import RerankingParser
    print('BLLIP imported OK')
except Exception as e:
    print(f'Import error: {e}')
    exit(1)

# Download and load the pre-trained WSJ model
print('Loading WSJ model (downloading if needed)...')
t0 = time.time()
try:
    rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
    print(f'Model loaded in {time.time()-t0:.1f}s')
except Exception as e:
    print(f'Model load failed: {e}')
    # Try alternate approach
    try:
        from bllipparser.ModelFetcher import download_model
        model_dir = download_model('WSJ-PTB3', '/tmp/bllip_models')
        rrp = RerankingParser.from_unified_model_dir(model_dir)
        print(f'Model loaded (alt) in {time.time()-t0:.1f}s')
    except Exception as e2:
        print(f'Alt load also failed: {e2}')
        exit(1)

# Test sentences of increasing length
test_sents = [
    "The dog saw the cat .",
    "Yields on mortgage-backed securities fell yesterday .",
    "The economy grew at a solid pace last quarter despite rising inflation .",
    "The committee voted to approve the budget after a long debate on spending priorities .",
    "The Federal Reserve said it would keep interest rates unchanged amid concerns about economic growth .",
]

print('\nParsing speed test:')
for sent in test_sents:
    words = sent.split()
    t1 = time.time()
    try:
        nbest = rrp.parse(sent)
        elapsed = time.time() - t1
        best = nbest[0]
        score = best.reranker_score if hasattr(best, 'reranker_score') else best.parser_score
        print(f'  words={len(words):2d}  score={score:.3f}  '
              f'time={elapsed:.3f}s  tree={str(best.ptb_parse)[:60]}')
    except Exception as e:
        print(f'  words={len(words):2d}  FAILED: {e}')
